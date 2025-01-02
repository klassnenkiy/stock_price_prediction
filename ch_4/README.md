# Stock Price Prediction with Linear Regression

## Описание

Этот проект предоставляет API и веб-приложение для предсказания цен акций с использованием линейной регрессии. Приложение позволяет загружать данные о ценах акций, обучать модели и получать прогнозы для разных тикеров.

## Структура проекта

- `backend/` — сервис для обучения моделей и прогнозирования.
- `frontend/` — интерфейс на Streamlit для взаимодействия с пользователем.
- `docker-compose.yml` — конфигурация для запуска всех сервисов (backend, frontend, Elasticsearch, Kibana, Filebeat).

## Установка и запуск

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/your-repo/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. Запустите проект:
    ```bash
    docker-compose up --build
    ```

3. Перейдите в браузер и откройте:
    - Backend API: `http://localhost:8000`
    - Frontend (Streamlit): `http://localhost:8502`

## Использование

### Веб-приложение (Frontend)

1. Загрузите CSV-файл с данными о ценах акций.
2. Выберите тикер, размер окна и количество дней для прогноза.
3. Нажмите кнопку "Обучить модель" для обучения модели.
4. Просматривайте прогнозы и метрики модели.

### API (Backend)

1. Используйте маршруты API для обучения модели, получения прогнозов и работы с моделями:
    - `POST /fit` — обучение модели.
    - `GET /predict` — прогноз для активной модели.
    - `POST /set` — установка активной модели.
    - `GET /models` — список всех моделей.
    - `GET /compare_models` — сравнение нескольких моделей.

## Пример использования

```bash
# Пример обучения модели для тикера SBER
curl -X 'POST' \
  'http://localhost:8000/fit?ticker=SBER' \
  -H 'Content-Type: application/json' \
  -d '{
  "window_size": 10,
  "forecast_days": 30
}'