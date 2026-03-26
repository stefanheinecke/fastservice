from data import Predictor
from enums import CloudProvider

cloud_provider = CloudProvider.GCP

if __name__ == "__main__":
    symbol = "GC=F"

    predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
    print(f"Storing predictions for {symbol}...")

    forecast_df, past_df, df = predict_obj.create_predictions()
    print(f"Forecast DataFrame for {symbol}:\n{forecast_df}")

    predict_obj.store_predictions(past_df)
    print(f"Stored Past Predictions for {symbol} in BigQuery.")

    predict_obj.store_predictions(forecast_df)
    print(f"Stored Future Predictions for {symbol} in BigQuery.")

    predict_obj.update_with_real_close(df)
    print("Prediction job completed.")
