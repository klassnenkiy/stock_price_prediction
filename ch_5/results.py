models_metrics = {
    "Random Forest": rf_improved_results,
    "Exponential Smoothing": exp_smoothing_improved_results,
    "Gradient Boosting": gb_improved_results,
    "Decision Tree": dt_improved_results,
    "XGBoost": xgb_improved_results,
    "LGBM": lgbm_improved_results,
    "k-NN": knn_improved_results
}

metrics_comparison = []
for model_name, results in models_metrics.items():
    for ticker, result in results.items():
        metrics_comparison.append({
            "Model": model_name,
            "Ticker": ticker,
            "MSE": result["Metrics"]["MSE"],
            "MAE": result["Metrics"]["MAE"],
            "RMSE": result["Metrics"]["RMSE"],
            "MAPE": result["Metrics"]["MAPE"],
            "R2": result["Metrics"]["R2"]
        })

metrics_df = pd.DataFrame(metrics_comparison)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
display(metrics_df)
