# Прогнозирование цен акций

### Описание проекта:

Проект **"Прогнозирование цен акций"** направлен на разработку системы прогнозирования рыночных цен акций с использованием методов машинного обучения (ML) и глубокого обучения (DL). Проект включает основные этапы работы:

1. **Разведочный анализ данных (EDA):** Сбор и анализ макроэкономических данных и финансовых показателей компаний. Оценка закономерностей, обработка выбросов и аномалий, формирование и тестирование гипотез ([Ноутбук](https://github.com/klassnenkiy/stock_price_prediction/blob/main/ch_2/EDA_Year_project_Tyulyagin_ch2.ipynb))

2. **Разработка моделей машинного обучения (ML):** Предобработка данных, создание моделей временных рядов, регрессий, XGBoost и других. Модели будут оцениваться на основе ошибок прогноза и точности, а также оптимизироваться с помощью гиперпараметров через пайплайны для улучшения точности прогнозов. ([Бейзлайн](https://github.com/klassnenkiy/stock_price_prediction/blob/main/ch_3/Baseline_ch3.ipynb))

3. **Модели глубокого обучения (DL):** Применение рекуррентных нейронных сетей LSTM и GRU для прогнозирования временных рядов. Сравнение точности между ML и DL моделями, будут применяться механизмы внимания и регуляризации, а также создан ансамбль моделей для повышения точности предсказаний. [LSTM-GRU](https://github.com/klassnenkiy/stock_price_prediction/tree/main/ch_6)


4. **Создание веб-сервиса:** разработка микросервисной архитектуры с использованием FastAPI (или аналога). Включает построение REST API для прогнозов, реализацию простого фронтенда для визуализации результатов и развертывание на облачной платформе с мониторингом точности прогнозов.

Проект ориентирован на интеграцию новейших технологий в области финансового прогнозирования и создание удобного для пользователя сервиса для анализа акций.

**Куратор:** Мария Кофанова (@miya_hse)


**Студент:**

- [Тюлягин Станислав Игоревич](https://github.com/klassnenkiy), ([@tyulyagins](https://t.me/tyulyagins))
