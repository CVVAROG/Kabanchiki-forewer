package main

import (
	"net/http"
	"strconv"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

// ==== Модели под JSON ====

type Topic struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

type TopicsResponse struct {
	Topics []Topic `json:"topics"`
}

type SentimentStats struct {
	Positive int `json:"positive"`
	Neutral  int `json:"neutral"`
	Negative int `json:"negative"`
}

type TopicsStatsItem struct {
	ID    int            `json:"id"`
	Name  string         `json:"name"`
	Stats SentimentStats `json:"stats"`
}

type Period struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type TopicsStatsResponse struct {
	Period Period            `json:"period"`
	Topics []TopicsStatsItem `json:"topics"`
}

type TimelinePoint struct {
	Date     string `json:"date"`
	Positive int    `json:"positive"`
	Neutral  int    `json:"neutral"`
	Negative int    `json:"negative"`
}

type TimelineResponse struct {
	Topic    Topic           `json:"topic"`
	Timeline []TimelinePoint `json:"timeline"`
}

type ReviewsFilters struct {
	TopicID   *int    `json:"topic_id,omitempty"`
	Sentiment *string `json:"sentiment,omitempty"`
	Period    *Period `json:"period,omitempty"`
}

type Pagination struct {
	Page  int `json:"page"`
	Limit int `json:"limit"`
	Total int `json:"total"`
}

type ReviewItem struct {
	ID        int    `json:"id"`
	Date      string `json:"date"`
	Sentiment string `json:"sentiment"`
	Text      string `json:"text"`
	Region    string `json:"region"`
}

type ReviewsResponse struct {
	Filters    ReviewsFilters `json:"filters"`
	Pagination Pagination     `json:"pagination"`
	Reviews    []ReviewItem   `json:"reviews"`
}

// ==== Хендлеры ====

func handleGetTopics(c echo.Context) error {
	topics := []Topic{
		{ID: 1, Name: "Ипотека"},
		{ID: 2, Name: "Карты"},
		{ID: 3, Name: "Кредиты"},
		{ID: 4, Name: "Вклады"},
	}
	return c.JSON(http.StatusOK, TopicsResponse{Topics: topics})
}

func handleGetTopicsStats(c echo.Context) error {
	from := c.QueryParam("date_from")
	to := c.QueryParam("date_to")
	_ = c.QueryParam("region") // опционально, просто игнорим в заглушке

	resp := TopicsStatsResponse{
		Period: Period{From: from, To: to},
		Topics: []TopicsStatsItem{
			{ID: 1, Name: "Ипотека", Stats: SentimentStats{Positive: 120, Neutral: 45, Negative: 35}},
			{ID: 2, Name: "Карты", Stats: SentimentStats{Positive: 200, Neutral: 80, Negative: 60}},
		},
	}
	return c.JSON(http.StatusOK, resp)
}

func handleGetTopicTimeline(c echo.Context) error {
	topicIDStr := c.Param("topic_id")
	topicID, _ := strconv.Atoi(topicIDStr)

	topic := Topic{ID: topicID, Name: "Топик " + strconv.Itoa(topicID)}
	switch topicID {
	case 1:
		topic.Name = "Ипотека"
	case 2:
		topic.Name = "Карты"
	case 3:
		topic.Name = "Кредиты"
	case 4:
		topic.Name = "Вклады"
	}

	// Поддерживаем параметры запроса (не влияют на мок)
	_ = c.QueryParam("date_from")
	_ = c.QueryParam("date_to")
	_ = c.QueryParam("group_by")
	_ = c.QueryParam("region")

	timeline := []TimelinePoint{
		{Date: "2024-01-01", Positive: 5, Neutral: 2, Negative: 1},
		{Date: "2024-01-02", Positive: 8, Neutral: 3, Negative: 4},
		{Date: "2024-01-03", Positive: 12, Neutral: 5, Negative: 2},
	}

	return c.JSON(http.StatusOK, TimelineResponse{Topic: topic, Timeline: timeline})
}

func handleGetReviews(c echo.Context) error {
	// Фильтры
	var topicIDPtr *int
	if tid := c.QueryParam("topic_id"); tid != "" {
		if v, err := strconv.Atoi(tid); err == nil {
			topicID := v
			topicIDPtr = &topicID
		}
	}

	var sentimentPtr *string
	if s := c.QueryParam("sentiment"); s != "" {
		sentiment := s
		sentimentPtr = &sentiment
	}

	from := c.QueryParam("date_from")
	to := c.QueryParam("date_to")

	var periodPtr *Period
	if from != "" && to != "" {
		periodPtr = &Period{From: from, To: to}
	}

	// Пагинация
	page := 1
	if p := c.QueryParam("page"); p != "" {
		if v, err := strconv.Atoi(p); err == nil && v > 0 {
			page = v
		}
	}
	limit := 20
	if l := c.QueryParam("limit"); l != "" {
		if v, err := strconv.Atoi(l); err == nil && v > 0 {
			limit = v
		}
	}

	resp := ReviewsResponse{
		Filters: ReviewsFilters{
			TopicID:   topicIDPtr,
			Sentiment: sentimentPtr,
			Period:    periodPtr,
		},
		Pagination: Pagination{Page: page, Limit: limit, Total: 247},
		Reviews: []ReviewItem{
			{ID: 9321, Date: "2024-01-03", Sentiment: "negative", Text: "Очень долго оформляется ипотека!", Region: "Москва"},
			{ID: 9322, Date: "2024-01-03", Sentiment: "negative", Text: "Банк затянул с одобрением заявки.", Region: "Санкт-Петербург"},
		},
	}

	return c.JSON(http.StatusOK, resp)
}

func main() {
	e := echo.New()
	e.HideBanner = true

	e.Use(middleware.Recover())
	e.Use(middleware.Logger())
	// Если фронт будет на другом домене/порту — раскомментируй:
	// e.Use(middleware.CORS())

	// API-эндпоинты
	e.GET("/topics", handleGetTopics)
	e.GET("/topics/stats", handleGetTopicsStats)
	e.GET("/topics/:topic_id/timeline", handleGetTopicTimeline)
	e.GET("/reviews", handleGetReviews)

	// Отдаём фронтенд из папки ./frontend (index.html будет на /)
	// Важно: статик вешаем ПОСЛЕ API, чтобы роуты не перебивались.
	e.Static("/", "frontend")

	e.Logger.Fatal(e.Start(":8080"))
}
