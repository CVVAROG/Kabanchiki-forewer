import logging
from typing import List, Dict, Tuple

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")

# ==== Константы (без ENV) ====
MODEL_PATH = "/app/final_model_opset19.onnx"
TOKENIZER_DIR = "/app"  # в контейнере это директория model/
PREFIX = "categorize_topic: "
MAX_LEN = 512
BATCH_SIZE = 16
THRESHOLD = 0.5

# Классы в правильном порядке (topic_sentiment)
CLASSES = [
    'Банкоматы_нейтрально', 'Банкоматы_отрицательно', 'Банкоматы_положительно',
    'Безопасность/блокировки_нейтрально', 'Безопасность/блокировки_отрицательно', 'Безопасность/блокировки_положительно',
    'Вклад_нейтрально', 'Вклад_отрицательно', 'Вклад_положительно',
    'Дебетовая карта_нейтрально', 'Дебетовая карта_отрицательно', 'Дебетовая карта_положительно',
    'Ипотека_нейтрально', 'Ипотека_отрицательно', 'Ипотека_положительно',
    'Комиссии и тарифы_нейтрально', 'Комиссии и тарифы_отрицательно', 'Комиссии и тарифы_положительно',
    'Кредитная карта_нейтрально', 'Кредитная карта_отрицательно', 'Кредитная карта_положительно',
    'Кэшбэк/бонусы_нейтрально', 'Кэшбэк/бонусы_отрицательно', 'Кэшбэк/бонусы_положительно',
    'Обслуживание (качество)_нейтрально', 'Обслуживание (качество)_отрицательно', 'Обслуживание (качество)_положительно',
    'Отделения_нейтрально', 'Отделения_отрицательно', 'Отделения_положительно',
    'Переводы и платежи_нейтрально', 'Переводы и платежи_отрицательно', 'Переводы и платежи_положительно',
    'Поддержка/колл-центр_нейтрально', 'Поддержка/колл-центр_отрицательно', 'Поддержка/колл-центр_положительно',
    'Потребкредит_нейтрально', 'Потребкредит_отрицательно', 'Потребкредит_положительно',
    'Приложение и сайт_нейтрально', 'Приложение и сайт_отрицательно', 'Приложение и сайт_положительно',
    'Прочее_нейтрально', 'Прочее_отрицательно', 'Прочее_положительно',
    'Реструктуризация/кредитные каникулы_нейтрально', 'Реструктуризация/кредитные каникулы_отрицательно', 'Реструктуризация/кредитные каникулы_положительно',
    'Условия и информация_нейтрально', 'Условия и информация_отрицательно', 'Условия и информация_положительно'
]

# ==== Подготовка маппинга topic -> sentiment -> class_idx ====
def build_topic_maps(classes: List[str]) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    topic_to_sent_idx: Dict[str, Dict[str, int]] = {}
    topics_order: List[str] = []
    for i, lab in enumerate(classes):
        topic, sentiment = lab.rsplit("_", 1)
        if topic not in topic_to_sent_idx:
            topic_to_sent_idx[topic] = {}
            topics_order.append(topic)
        topic_to_sent_idx[topic][sentiment] = i
    return topic_to_sent_idx, topics_order

TOPIC_TO_SENT_IDX, TOPICS_ORDER = build_topic_maps(CLASSES)
SENTIMENTS_ORDER = ["нейтрально", "отрицательно", "положительно"]  # ожидаемые значения в CLASSES

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# ==== Pydantic v2 схемы ====
class Item(BaseModel):
    id: int = Field(..., description="Уникальный идентификатор отзыва")
    text: str = Field(..., min_length=1, description="Текст отзыва (UTF-8)")

class PredictRequest(BaseModel):
    data: List[Item] = Field(..., min_length=1, max_length=250)

class Prediction(BaseModel):
    id: int
    topics: List[str]
    sentiments: List[str]

class PredictResponse(BaseModel):
    predictions: List[Prediction]

# ==== FastAPI ====
app = FastAPI(
    title="ONNX Multilabel Topics+Sentiment API",
    version="3.0.0",
    description="Классификация тем и тональностей (мульти-лейбл) через ONNX. POST /predict"
)

session: ort.InferenceSession = None
tokenizer = None
output_names: List[str] = []
logits_name: str = "logits"

@app.on_event("startup")
def _startup():
    global session, tokenizer, output_names, logits_name

    if AutoTokenizer is None:
        raise RuntimeError("transformers не установлен — нужен для токенизации.")

    # Локальный токенайзер из /app (tokenizer.json, tokenizer_config.json, vocab.txt)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    logger.info("Tokenizer loaded from %s", TOKENIZER_DIR)

    # ONNX с CPU провайдером
    so = ort.SessionOptions()
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=so)

    output_names = [o.name for o in session.get_outputs()]
    # Предпочмем выход, содержащий "logit"
    cand = [n for n in output_names if "logit" in n.lower()] or output_names
    logits_name = cand[0]

    # Проверка входов
    input_names = [i.name for i in session.get_inputs()]
    required = {"input_ids", "attention_mask"}
    if not required.issubset(set(input_names)):
        raise RuntimeError(f"Ожидались входы {required}, но найдены: {input_names}")

    logger.info("Model inputs: %s, outputs: %s, logits: %s", input_names, output_names, logits_name)

@app.get("/health")
def health():
    return {"status": "ok"}

def infer_batch(texts: List[str]) -> np.ndarray:
    enc = tokenizer(
        [PREFIX + t for t in texts],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np"
    )
    feed = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    outputs = session.run(None, feed)
    out_map = {name: value for name, value in zip(output_names, outputs)}
    logits = out_map[logits_name]  # ожидаем [B, num_classes]
    return logits

def select_topics_and_sentiments(probs: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    probs: [num_classes] — вероятности после сигмоиды
    1) Берём по каждому топику лучшую тональность (max по трём классам topic_*).
    2) Оставляем только те топики, где лучшая вероятность >= THRESHOLD.
    3) Если ни один не прошёл порог — берём глобальный argmax (одну пару topic+sentiment).
    4) Сортируем выбранные топики по убыванию вероятности (стабильно).
    """
    best_for_topic: Dict[str, Tuple[str, float]] = {}  # topic -> (best_sentiment, prob)
    for topic, sent_to_idx in TOPIC_TO_SENT_IDX.items():
        sent_probs = [(sent, probs[idx]) for sent, idx in sent_to_idx.items()]
        sent_probs.sort(key=lambda x: x[1], reverse=True)
        best_for_topic[topic] = (sent_probs[0][0], float(sent_probs[0][1]))

    # Фильтрация по порогу
    chosen = [(topic, sent, p) for topic, (sent, p) in best_for_topic.items() if p >= THRESHOLD]

    if not chosen:
        # Фолбек: глобальный argmax по всем классам
        best_idx = int(np.argmax(probs))
        topic, sent = CLASSES[best_idx].rsplit("_", 1)
        return [topic], [sent]

    # Сортируем по вероятности убыв.
    chosen.sort(key=lambda x: x[2], reverse=True)

    topics = [t for (t, s, p) in chosen]
    sentiments = [s for (t, s, p) in chosen]
    return topics, sentiments

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.data) > 250:
        raise HTTPException(status_code=400, detail="В одном запросе допускается максимум 250 отзывов.")

    predictions: List[Dict] = []
    for i in range(0, len(req.data), BATCH_SIZE):
        batch = req.data[i : i + BATCH_SIZE]
        texts = [x.text for x in batch]
        ids = [x.id for x in batch]

        logits = infer_batch(texts)              # [B, C]
        probs = sigmoid(logits)                  # [B, C]

        for rid, p in zip(ids, probs):
            topics, sentiments = select_topics_and_sentiments(p)
            predictions.append({"id": rid, "topics": topics, "sentiments": sentiments})

    return {"predictions": predictions}