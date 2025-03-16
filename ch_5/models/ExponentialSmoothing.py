from statsmodels.tsa.holtwinters import ExponentialSmoothing


def exponential_smoothing_rolling_with_backtest_improved(ticker_data, window_size=20, test_forecast_days=30, future_forecast_days=30,
                                                         trend='add', seasonal=None, seasonal_periods=None, noise_scale=1.0):
    ticker_data = ticker_data.sort_values(by='TRADEDATE')
    series = ticker_data['CLOSE'].values
    dates = ticker_data['TRADEDATE'].values


    if len(series) <= window_size + test_forecast_days:
        print(f"Недостаточно данных для backtest-прогноза для тикера {ticker_data['TICKER'].iloc[0]}.")
        return None

    test_forecast = []
    test_forecast_dates = dates[-test_forecast_days:]

    for i in range(len(series) - test_forecast_days, len(series)):
        window_data = series[i - window_size:i]
        try:
            model = ExponentialSmoothing(window_data, trend=trend, seasonal=seasonal,
                                         seasonal_periods=seasonal_periods, initialization_method='estimated')
            fit_model = model.fit(optimized=True)
            pred = fit_model.forecast(steps=1)[0]
        except Exception as e:
            print(f"Ошибка при backtest-прогнозировании для индекса {i}: {e}")
            pred = np.nan
        test_forecast.append(pred)


    actual_test = series[-test_forecast_days:]
    mse = mean_squared_error(actual_test, test_forecast)
    mae = mean_absolute_error(actual_test, test_forecast)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual_test, test_forecast)
    r2 = r2_score(actual_test, test_forecast)
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


    residuals = np.array(actual_test) - np.array(test_forecast)
    residuals = residuals[~np.isnan(residuals)]
    error_std = np.std(residuals)


    future_forecast = []
    series_history = series.copy()
    for _ in range(future_forecast_days):
        window_data = series_history[-window_size:]
        try:
            model = ExponentialSmoothing(window_data, trend=trend, seasonal=seasonal,
                                         seasonal_periods=seasonal_periods, initialization_method='estimated')
            fit_model = model.fit(optimized=True)
            pred = fit_model.forecast(steps=1)[0]
        except Exception as e:
            print(f"Ошибка при форвард-прогнозировании: {e}")
            pred = np.nan

        noise = np.random.normal(loc=0, scale=error_std * noise_scale)
        pred_noisy = pred + noise
        future_forecast.append(pred_noisy)
        series_history = np.append(series_history, pred_noisy)

    future_forecast_dates = pd.date_range(dates[-1], periods=future_forecast_days+1, freq='B')[1:]

    plt.figure(figsize=(20, 6))
    plt.plot(dates, series, label='Actual Prices', color='blue', linestyle='-')
    plt.plot(test_forecast_dates, test_forecast, label='Backtest Forecast', color='green', linestyle='-')
    plt.plot(future_forecast_dates, future_forecast, label='Future Forecast (Improved)', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Exponential Smoothing Rolling Forecast (Improved) for {ticker_data["TICKER"].iloc[0]}\n'
              f'Backtest Metrics: MSE={mse:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape*100:.2f}%, R²={r2:.2f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "Test Forecast Dates": test_forecast_dates,
        "Test Forecast Prices": test_forecast,
        "Future Forecast Dates": future_forecast_dates,
        "Future Forecast Prices": future_forecast,
        "Metrics": metrics,
        "Noise Std": error_std * noise_scale
    }

exp_smoothing_improved_results = {}
for ticker in first_tier_tickers:
    ticker_data = data[data['TICKER'] == ticker]
    result = exponential_smoothing_rolling_with_backtest_improved(ticker_data, window_size=20, test_forecast_days=30,
                                                                  future_forecast_days=30, trend='add', seasonal=None,
                                                                  noise_scale=1.0)
    if result is not None:
        exp_smoothing_improved_results[ticker] = result
