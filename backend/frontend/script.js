//   Глобальные переменные для графиков
let sentimentChart = null; // график общего распределения отзывов
let trendsChart = null; // график динамики

//   DOM-элементы: список тем
const topicsContainer = document.getElementById("topics-container");
const topicsLoading = document.getElementById("topics-loading");
const topicsError = document.getElementById("topics-error");

//   DOM-элементы: статистика
const statisticsContent = document.getElementById("statistics-content");
const statisticsLoading = document.getElementById("statistics-loading");
const statisticsError = document.getElementById("statistics-error");
const statisticsPlaceholder = document.getElementById("statistics-placeholder");

//   DOM-элементы: счетчики отзывов
const totalReviews = document.getElementById("total-reviews");
const positiveCount = document.getElementById("positive-count");
const positiveBar = document.getElementById("positive-bar");
const positivePercent = document.getElementById("positive-percent");
const neutralCount = document.getElementById("neutral-count");
const neutralBar = document.getElementById("neutral-bar");
const neutralPercent = document.getElementById("neutral-percent");
const negativeCount = document.getElementById("negative-count");
const negativeBar = document.getElementById("negative-bar");
const negativePercent = document.getElementById("negative-percent");

//   DOM-элементы: динамика (тренды)
const trendsContent = document.getElementById("trends-content");
const trendsLoading = document.getElementById("trends-loading");
const trendsError = document.getElementById("trends-error");
const trendsPlaceholder = document.getElementById("trends-placeholder");

//   DOM-элементы: кнопки и селекты
const applyFiltersBtn = document.getElementById("apply-filters");
const resetSelectionBtn = document.getElementById("reset-selection");
const chartTypeSelect = document.getElementById("chart-type");

//   DOM-элементы: отзывы
const reviewsContainer = document.getElementById("reviews-container");
const reviewsLoading = document.getElementById("reviews-loading");
const reviewsError = document.getElementById("reviews-error");
const reviewsPagination = document.getElementById("reviews-pagination");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const currentPageSpan = document.getElementById("current-page");

//   DOM-элементы: загрузка JSON
const jsonFileNameSpan = document.getElementById("json-file-name");
const fileInput = document.getElementById("json-file-input");
const detachJsonBtn = document.getElementById("detach-json");
const downloadFileBtn = document.getElementById("download-btn");

//   Состояние
let selectedTopics = []; // выбранные темы пользователем

//   Пагинация отзывов
let currentPage = 1;
const reviewsPerPage = 10;
let totalPages = 1;

//   Универсальная загрузка данных
async function fetchData(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Ошибка сети: ${res.status}`);
  return await res.json();
}

//   Основная инициализация
document.addEventListener("DOMContentLoaded", () => {
  feather.replace(); // замена иконок feather на svg
  loadTopics(); // загрузка списка тем при старте

  // Установка дат по умолчанию (за последние 6 месяцев)
  const endDate = new Date();
  const startDate = new Date();
  startDate.setMonth(endDate.getMonth() - 6);

  ["start-date", "end-date"].forEach((id, idx) => {
    const input = document.getElementById(id);
    const d = idx === 0 ? startDate : endDate;
    input.value = d.toLocaleDateString("ru-RU");

    // автоформат даты: дд.мм.гггг
    input.addEventListener("input", (e) => {
      let value = e.target.value.replace(/\D/g, "").slice(0, 8);
      if (value.length >= 5) {
        e.target.value = `${value.slice(0, 2)}.${value.slice(
          2,
          4
        )}.${value.slice(4)}`;
      } else if (value.length >= 3) {
        e.target.value = `${value.slice(0, 2)}.${value.slice(2)}`;
      } else {
        e.target.value = value;
      }
    });
  });

  // Кнопка применения фильтров
  applyFiltersBtn.addEventListener("click", () => {
    if (selectedTopics.length === 0) {
      alert("Выберите хотя бы одну тему!");
      return;
    }
    alert("Фильтры пока в разработке"); // уведомление
    loadStatisticsForMultiple(selectedTopics);
    loadTrends(selectedTopics);
    loadReviews(selectedTopics, 1);
  });

  // Кнопка сброса выбора
  resetSelectionBtn.addEventListener("click", clearSelection);

  // Смена типа диаграммы
  chartTypeSelect.addEventListener("change", () => {
    if (selectedTopics.length > 0) loadStatisticsForMultiple(selectedTopics);
  });

  // Работа с файлами JSON
  fileInput.addEventListener("change", handleFileUpload);
  detachJsonBtn.addEventListener("click", detachJsonFile);
  downloadFileBtn.addEventListener("click", downloadJsonFile);
});

//   Очистка выбора тем
function clearSelection() {
  selectedTopics = [];
  document
    .querySelectorAll(".topic-card")
    .forEach((el) => el.classList.remove("active"));
  resetSelectionBtn.classList.add("hidden");

  statisticsPlaceholder.classList.remove("hidden");
  statisticsContent.classList.add("hidden");

  trendsContent.classList.add("hidden");
  trendsPlaceholder.classList.remove("hidden");

  reviewsContainer.innerHTML = "";
  reviewsPagination.classList.add("hidden");

  if (sentimentChart) {
    sentimentChart.destroy();
    sentimentChart = null;
  }
  if (trendsChart) {
    trendsChart.destroy();
    trendsChart = null;
  }
}

//   Загрузка списка тем
async function loadTopics() {
  topicsLoading.classList.remove("hidden");
  topicsError.classList.add("hidden");
  topicsContainer.classList.add("hidden");

  try {
    const data = await fetchData("/topics");
    renderTopics(data.topics);
    topicsLoading.classList.add("hidden");
    topicsContainer.classList.remove("hidden");
  } catch {
    topicsLoading.classList.add("hidden");
    topicsError.classList.remove("hidden");
  }
}

//   Отрисовка списка тем
function renderTopics(topics) {
  topicsContainer.innerHTML = "";

  topics.forEach((topic) => {
    const topicElement = document.createElement("div");
    topicElement.className = "topic-card button cursor-pointer";
    topicElement.innerHTML = `
      <div class="flex items-center">
        <i data-feather="file-text" class="mr-3 text-gray-100"></i>
        <span class="font-medium">${topic.name}</span>
      </div>
    `;

    // Клик по теме
    topicElement.addEventListener("click", () => {
      if (selectedTopics.find((t) => t.id === topic.id)) {
        // если тема уже выбрана → снять выбор
        selectedTopics = selectedTopics.filter((t) => t.id !== topic.id);
        topicElement.classList.remove("active");
      } else {
        // если тема не выбрана → выбрать
        selectedTopics.push(topic);
        topicElement.classList.add("active");
      }

      if (selectedTopics.length === 0) {
        clearSelection();
      } else {
        resetSelectionBtn.classList.remove("hidden");
        loadStatisticsForMultiple(selectedTopics);
        loadTrends(selectedTopics);
        loadReviews(selectedTopics, 1);
      }
    });

    topicsContainer.appendChild(topicElement);
  });

  feather.replace();
}

//   Загрузка статистики по нескольким темам
async function loadStatisticsForMultiple(topics) {
  statisticsContent.classList.remove("hidden");
  statisticsError.classList.add("hidden");
  statisticsPlaceholder.classList.add("hidden");
  statisticsLoading.classList.remove("hidden");

  if (sentimentChart) sentimentChart.destroy();

  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;

  try {
    const data = await fetchData(
      `/topics/stats?date_from=${startDate}&date_to=${endDate}`
    );
    const filtered = data.topics.filter((t) =>
      topics.find((sel) => sel.id === t.id)
    );
    statisticsLoading.classList.add("hidden");
    renderMultiStatistics(filtered);
  } catch {
    statisticsLoading.classList.add("hidden");
    statisticsError.classList.remove("hidden");
  }
}

//   Отрисовка статистики
function renderMultiStatistics(topicsStats) {
  const total = topicsStats.reduce(
    (sum, t) => sum + t.stats.positive + t.stats.neutral + t.stats.negative,
    0
  );

  if (total === 0) {
    statisticsContent.classList.add("hidden");
    statisticsPlaceholder.classList.remove("hidden");
    if (sentimentChart) {
      sentimentChart.destroy();
      sentimentChart = null;
    }
    return;
  }

  const positive = topicsStats.reduce((sum, t) => sum + t.stats.positive, 0);
  const neutral = topicsStats.reduce((sum, t) => sum + t.stats.neutral, 0);
  const negative = topicsStats.reduce((sum, t) => sum + t.stats.negative, 0);

  // отображение цифр
  totalReviews.textContent = total;
  positiveCount.textContent = positive;
  neutralCount.textContent = neutral;
  negativeCount.textContent = negative;

  // проценты
  const positivePct = Math.round((positive / total) * 100);
  const neutralPct = Math.round((neutral / total) * 100);
  const negativePct = Math.round((negative / total) * 100);

  positiveBar.style.width = `${positivePct}%`;
  positivePercent.textContent = `${positivePct}%`;
  neutralBar.style.width = `${neutralPct}%`;
  neutralPercent.textContent = `${neutralPct}%`;
  negativeBar.style.width = `${negativePct}%`;
  negativePercent.textContent = `${negativePct}%`;

  // диаграмма
  const chartType = chartTypeSelect.value;
  const ctx = document.getElementById("sentimentChart").getContext("2d");
  sentimentChart = new Chart(ctx, {
    type: chartType,
    data: {
      labels: [
        "Положительные отзывы",
        "Нейтральные отзывы",
        "Отрицательные отзывы",
      ],
      datasets: [
        {
          data: [positive, neutral, negative],
          backgroundColor: ["#10B981", "#F59E0B", "#EF4444"],
          borderColor: ["#065F46", "#78350F", "#991B1B"],
          hoverBackgroundColor: ["#0e8451ff", "#bc891cff", "#ce2029"],
          borderWidth: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }, // скрываем стандартную легенду
      },
    },
  });
}

//   Загрузка графика динамики
async function loadTrends(topics) {
  trendsContent.classList.add("hidden");
  trendsError.classList.add("hidden");
  trendsPlaceholder.classList.add("hidden");
  trendsLoading.classList.remove("hidden");

  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;

  try {
    const datasets = [];
    const labelsSet = new Set();

    // цвета для разных типов отзывов
    const sentimentColors = {
      positive: "#10B981",
      neutral: "#F59E0B",
      negative: "#EF4444",
    };

    // формируем набор данных по каждой теме
    for (const topic of topics) {
      const data = await fetchData(
        `/topics/${topic.id}/timeline?date_from=${startDate}&date_to=${endDate}&group_by=day`
      );

      if (data.timeline && data.timeline.length > 0) {
        data.timeline.forEach((p) => labelsSet.add(p.date));

        datasets.push({
          label: `${topic.name} — Положительные`,
          topic: topic.name,
          sentiment: "positive",
          data: data.timeline.map((p) => p.positive),
          borderColor: sentimentColors.positive,
          backgroundColor: sentimentColors.positive + "33",
          fill: false,
        });
        datasets.push({
          label: `${topic.name} — Нейтральные`,
          topic: topic.name,
          sentiment: "neutral",
          data: data.timeline.map((p) => p.neutral),
          borderColor: sentimentColors.neutral,
          backgroundColor: sentimentColors.neutral + "33",
          fill: false,
        });
        datasets.push({
          label: `${topic.name} — Отрицательные`,
          topic: topic.name,
          sentiment: "negative",
          data: data.timeline.map((p) => p.negative),
          borderColor: sentimentColors.negative,
          backgroundColor: sentimentColors.negative + "33",
          fill: false,
        });
      }
    }

    const labels = Array.from(labelsSet).sort();

    if (datasets.length === 0) {
      trendsLoading.classList.add("hidden");
      trendsPlaceholder.classList.remove("hidden");
      if (trendsChart) {
        trendsChart.destroy();
        trendsChart = null;
      }
      return;
    }

    // создаём график
    if (trendsChart) trendsChart.destroy();
    const ctx = document.getElementById("trendsChart").getContext("2d");
    trendsChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false, // отключаем стандартную легенду
          },
        },
        scales: { y: { beginAtZero: true } },
      },
    });

    //   Кастомная легенда
    const legendContainerId = "custom-trends-legend";
    let legendContainer = document.getElementById(legendContainerId);
    if (!legendContainer) {
      legendContainer = document.createElement("div");
      legendContainer.id = legendContainerId;
      legendContainer.className = "mt-6 flex justify-center space-x-12 text-sm";
      document.getElementById("trends-content").appendChild(legendContainer);
    }
    legendContainer.innerHTML = "";

    const sentiments = ["positive", "neutral", "negative"];
    const sentimentLabels = {
      positive: "Положительные",
      neutral: "Нейтральные",
      negative: "Отрицательные",
    };

    sentiments.forEach((sentiment) => {
      const group = document.createElement("div");
      group.className = "flex flex-col items-center";

      // Заголовок группы
      const header = document.createElement("div");
      header.className =
        "font-medium flex items-center space-x-2 cursor-pointer mb-2";
      header.dataset.active = "true";

      const colorBox = document.createElement("span");
      colorBox.style.backgroundColor = sentimentColors[sentiment];
      colorBox.className = "inline-block w-3 h-3 rounded";

      const label = document.createElement("span");
      label.textContent = sentimentLabels[sentiment];

      header.appendChild(colorBox);
      header.appendChild(label);
      group.appendChild(header);

      // Контейнер для тем
      const topicsWrapper = document.createElement("div");
      topicsWrapper.className = "flex flex-wrap gap-2 justify-center";

      topics.forEach((topic) => {
        const dsIndex = datasets.findIndex(
          (d) => d.sentiment === sentiment && d.topic === topic.name
        );
        if (dsIndex === -1) return;

        const item = document.createElement("span");
        item.className = "cursor-pointer";
        item.textContent = topic.name;
        item.dataset.active = "true";

        // Клик по конкретной теме
        item.addEventListener("click", () => {
          const active = item.dataset.active === "true";
          item.dataset.active = active ? "false" : "true";
          item.style.textDecoration = active ? "line-through" : "none";
          trendsChart.setDatasetVisibility(dsIndex, !active);
          trendsChart.update();

          // проверка: все ли темы отключены?
          const allTopics = topicsWrapper.querySelectorAll("span");
          const allInactive = Array.from(allTopics).every(
            (el) => el.dataset.active === "false"
          );
          header.style.textDecoration = allInactive ? "line-through" : "none";
          header.dataset.active = allInactive ? "false" : "true";
        });

        topicsWrapper.appendChild(item);
      });

      // Клик по заголовку → отключает/включает все темы
      header.addEventListener("click", () => {
        const active = header.dataset.active === "true";
        header.dataset.active = active ? "false" : "true";
        header.style.textDecoration = active ? "line-through" : "none";

        datasets.forEach((ds, i) => {
          if (ds.sentiment === sentiment) {
            trendsChart.setDatasetVisibility(i, !active);
          }
        });
        trendsChart.update();

        topicsWrapper.querySelectorAll("span").forEach((item) => {
          item.dataset.active = active ? "false" : "true";
          item.style.textDecoration = active ? "line-through" : "none";
        });
      });

      group.appendChild(topicsWrapper);
      legendContainer.appendChild(group);
    });

    trendsLoading.classList.add("hidden");
    trendsContent.classList.remove("hidden");
  } catch {
    trendsLoading.classList.add("hidden");
    trendsError.classList.remove("hidden");
  }
}

//   Загрузка отзывов
async function loadReviews(topics, page = 1) {
  reviewsContainer.innerHTML = "";
  reviewsLoading.classList.remove("hidden");
  reviewsError.classList.add("hidden");

  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;

  try {
    let allReviews = [];
    for (const topic of topics) {
      const data = await fetchData(
        `/reviews?topic_id=${topic.id}&date_from=${startDate}&date_to=${endDate}`
      );
      if (data.reviews && data.reviews.length > 0) {
        allReviews = allReviews.concat(data.reviews);
      }
    }

    reviewsLoading.classList.add("hidden");

    if (allReviews.length === 0) {
      reviewsContainer.innerHTML = `<p class="text-gray-500">Нет отзывов за выбранный период</p>`;
      reviewsPagination.classList.add("hidden");
      return;
    }

    // сортировка по дате
    allReviews.sort((a, b) => new Date(b.date) - new Date(a.date));

    totalPages = Math.ceil(allReviews.length / reviewsPerPage);
    currentPage = Math.min(page, totalPages);
    const startIdx = (currentPage - 1) * reviewsPerPage;
    const endIdx = startIdx + reviewsPerPage;
    const pageReviews = allReviews.slice(startIdx, endIdx);

    renderReviews(pageReviews);

    // пагинация
    reviewsPagination.classList.remove("hidden");
    currentPageSpan.textContent = `${currentPage} / ${totalPages}`;
    prevPageBtn.disabled = currentPage === 1;
    nextPageBtn.disabled = currentPage === totalPages;

    prevPageBtn.onclick = () => {
      if (currentPage > 1) loadReviews(topics, currentPage - 1);
    };
    nextPageBtn.onclick = () => {
      if (currentPage < totalPages) loadReviews(topics, currentPage + 1);
    };
  } catch {
    reviewsLoading.classList.add("hidden");
    reviewsError.classList.remove("hidden");
  }
}

//   Отрисовка отзывов
function renderReviews(reviews) {
  reviewsContainer.innerHTML = "";
  reviews.forEach((review) => {
    const reviewEl = document.createElement("div");
    reviewEl.className = "p-4 mb-3 border rounded-lg bg-white shadow-sm";
    reviewEl.innerHTML = `
      <div class="flex justify-between mb-2">
        <span class="text-sm text-gray-500">${review.date} | ${
      review.region || "Регион не указан"
    }</span>
        <span class="px-2 py-1 rounded text-xs ${
          review.sentiment === "positive"
            ? "bg-green-100 text-green-700"
            : review.sentiment === "neutral"
            ? "bg-yellow-100 text-yellow-700"
            : "bg-red-100 text-red-700"
        }">${review.sentiment}</span>
      </div>
      <p class="text-gray-800">${review.text}</p>
    `;
    reviewsContainer.appendChild(reviewEl);
  });
}

//   Работа с JSON-файлом

// Загрузка JSON
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error("Ошибка загрузки файла");
      detachJsonBtn.classList.remove("hidden");
      downloadFileBtn.classList.remove("hidden");
      jsonFileNameSpan.textContent = file.name;
    })
    .catch(() => alert("Ошибка при отправке JSON файла"));
}

// Открепление JSON
function detachJsonFile() {
  fileInput.value = "";
  detachJsonBtn.classList.add("hidden");
  downloadFileBtn.classList.add("hidden");
  jsonFileNameSpan.textContent = "Файл еще не выбран";
}

// Скачивание JSON
function downloadJsonFile() {
  window.location.href = "/download";
}
