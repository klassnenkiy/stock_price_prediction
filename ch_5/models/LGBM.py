from lightgbm import LGBMRegressor


def lgbm_rolling_with_backtest_improved(ticker_data, window_size=20, test_forecast_days=30, future_forecast_days=30, noise_scale=1.0):

    ticker_data = ticker_data.dropna(subset=['CLOSE', 'TRADEDATE'])

    ticker_data = ticker_data.sort_values(by='TRADEDATE')
    series = ticker_data['CLOSE'].values
    dates = ticker_data['TRADEDATE'].values


    if len(series) <= window_size + test_forecast_days:
        print(f"Недостаточно данных для backtest-прогноза для тикера {ticker_data['TICKER'].iloc[0]}.")
        return None


    def create_dataset(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)


    test_forecast = []
    test_forecast_dates = dates[-test_forecast_days:]

    for i in range(len(series) - test_forecast_days, len(series)):
        if i < window_size:
            test_forecast.append(np.nan)
            continue
        X_train, y_train = create_dataset(series[:i], window_size)
        if len(X_train) == 0:
            test_forecast.append(np.nan)
            continue
        model = LGBMRegressor(random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        X_pred = series[i-window_size:i].reshape(1, -1)
        pred = model.predict(X_pred)[0]
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
        X_train, y_train = create_dataset(series_history, window_size)
        if len(X_train) == 0:
            future_forecast.append(np.nan)
            continue
        model = LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)
        X_pred = series_history[-window_size:].reshape(1, -1)
        pred = model.predict(X_pred)[0]

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
    plt.title(f'LGBM Rolling Forecast (Improved) for {ticker_data["TICKER"].iloc[0]}\n'
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

lgbm_improved_results = {}
for ticker in first_tier_tickers:
    ticker_data = data[data['TICKER'] == ticker]
    result = lgbm_rolling_with_backtest_improved(ticker_data, window_size=20, test_forecast_days=30, future_forecast_days=30, noise_scale=1.0)
    if result is not None:
        lgbm_improved_results[ticker] = result
