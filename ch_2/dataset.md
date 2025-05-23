# Описание данных для проекта "Прогнозирование цен акций"

## Общая информация

- **Название проекта**: Прогнозирование цен акций
- **Источник данных**: Московская биржа (MOEX) через API ISS ([https://iss.moex.com](https://iss.moex.com))
- **Количество акций (тикеров)**: 250
- **Таймфрейм данных**: Дневной (с 2023-10-30 по 2024-10-30)
- **Формат данных**: Табличные данные в формате CSV
- **Ссылка на данные**: [Ссылка с CSV-файлом](https://drive.google.com/file/d/1Rn3-XWfgK-fs7-8G2HM9bkLQ9utTA4wJ/view?usp=sharing)

## Описание признаков

Каждая строка датасета представляет данные по одной акции на одну дату. Признаки включают следующую информацию:

| Признак       | Тип данных    | Описание                                                                                   |
|---------------|---------------|--------------------------------------------------------------------------------------------|
| `TRADEDATE`   | `datetime`    | Дата торговой сессии в формате `YYYY-MM-DD`                                                |
| `OPEN`        | `float`       | Цена открытия акции на начало торгового дня                                                |
| `HIGH`        | `float`       | Максимальная цена акции за день                                                            |
| `LOW`         | `float`       | Минимальная цена акции за день                                                             |
| `CLOSE`       | `float`       | Цена закрытия акции в конце торгового дня                                                  |
| `VOLUME`      | `int`         | Объем торгов по данной акции за день (число проданных акций)                               |
| `TICKER`      | `string`      | Уникальный тикер акции (идентификатор на бирже, например, `GAZP` для Газпрома)             |

## Количество объектов по каждому тикеру
Общее количество данных и структура

Анализ был проведён на основании набора данных, содержащего информацию по торговле акциями, включая следующие поля:

- **TRADEDATE**: Дата торгов.
- **OPEN**: Цена открытия.
- **HIGH**: Максимальная цена.
- **LOW**: Минимальная цена.
- **CLOSE**: Цена закрытия.
- **VOLUME**: Объём торгов.
- **TICKER**: Символ акции.

- **Количество записей для каждого тикера**: варьируется в зависимости от торговой активности (в среднем около 200 торговых дней за год).
- **Общий объем записей**: **63,082 записи**, охватывающих **250 уникальных тикеров**.
- **Период данных**: С октября 2023 года по октябрь 2024 года.

Набор данных включает 250 уникальных тикеров, что позволяет анализировать разнообразные акции и их поведение на российском рынке.

## Информация о пропущенных значениях

### Обнаруженные пропуски
Для некоторых тикеров могут отсутствовать данные за отдельные дни. Причины:
- **Нерабочие дни** на бирже (например, выходные и праздничные дни).
- **Отсутствие торгов по конкретным акциям** в определенные дни.
  
### Заполнение пропусков
- Для тикеров с отсутствующими значениями используются следующие методы:
    - **Заполнение средними значениями**: в случае пропусков по значению `CLOSE` можно интерполировать данные на основе предыдущих значений.
    - **Удаление пропусков**: если отсутствуют данные за весь торговый день, строки удаляются.

- **Общее количество пропусков:**
  - В столбцах `OPEN`, `HIGH`, `LOW`, и `CLOSE` по 1655 пропусков в каждом (по 2.62%).

- **Вывод:**
 Пропуски в ценах могут повлиять на статистические вычисления и модели.

## Примечания

1. **Типичные филлеры**:
   - В случае недоступных данных или нерабочих дней по акциям, ячейки остаются пустыми (`NaN`).
2. **Аномалии и выбросы**:
   - Выбросы могут включать случаи, когда объем торгов (`VOLUME`) равен нулю. Такие данные нужно учитывать при анализе аномалий.
3. **Целевой признак**:
   - Признак `CLOSE` может использоваться в качестве целевого признака для прогнозирования цены акции.


## Основные статистические показатели

- **Цены (OPEN, HIGH, LOW, CLOSE):**
  - Средние значения колеблются от 2343.49 до 2381.42.
  - Наблюдаются значительные стандартные отклонения, указывающие на высокую волатильность цен (например, максимальное значение CLOSE — 166850).

- **Объем торгов (VOLUME):**
  - Средний объем торгов составляет около 402,113,700, с максимальным значением до 461,548,200,000.

## Волатильность акций первого эшелона

- **Наибольшая волатильность:** 
  - PLZL (ПАО Полюс) — 1183.07, также показывает высокую волатильность.

- **Низкая волатильность:**
  - Акции SNGS (Сургутнефтегаз) и SNGSP имеют самую низкую волатильность — 3.07 и 6.82 соответственно, что указывает на более стабильные цены.

## Корреляция между акциями

Коэффициенты корреляции показывают, что:
- CHMF и GAZP имеют положительную корреляцию (0.257), что может указывать на некоторую степень зависимости в движениях цен.
- GMKN и NVTK имеют высокую корреляцию с другими тикерами, что может свидетельствовать о том, что их поведение может быть связано с общими рыночными факторами.


## Выводы по датасету

Данный датасет предоставляет достаточную информацию для анализа временных рядов цен акций и выполнения прогнозов на основе этих данных. Использование различных признаков (`OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOLUME`) позволяет строить модели с учетом внутридневных колебаний и общего тренда. В случае обнаружения дополнительных характеристик или выявления новых аномалий в процессе анализа файл будет обновлен. Впоследствии датасет будет и может дополняться новыми данными и признаками.

---
Скрипт для сбора данных https://github.com/klassnenkiy/stock_price_prediction/blob/main/parser.ipynb


**Дата обновления файла**: `2024-11-02`

