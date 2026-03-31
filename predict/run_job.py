import os
import sys
from data import Predictor
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_all_symbols(database_url):
    """Fetch distinct symbols that already have predictions in the DB."""
    engine = create_engine(database_url)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM predictions ORDER BY symbol"))
        return [r[0] for r in rows]


def run_for_symbol(database_url, symbol):
    print(f"\n{'='*50}")
    print(f"Processing {symbol}...")
    print(f"{'='*50}")
    predict_obj = Predictor(database_url, symbol)
    forecast_df, past_df, df = predict_obj.create_predictions()
    print(f"Forecast DataFrame for {symbol}:\n{forecast_df}")
    predict_obj.store_predictions(past_df)
    print(f"Stored Past Predictions for {symbol}.")
    predict_obj.store_predictions(forecast_df)
    print(f"Stored Future Predictions for {symbol}.")
    predict_obj.update_with_real_close(df)

    # Fetch and store stats for this symbol
    import requests
    try:
        api_url = f"http://localhost:5000/api/summary-predictions/{symbol}"
        print(f"Fetching summary stats for {symbol} from {api_url} ...")
        resp = requests.get(api_url)
        resp.raise_for_status()
        stat = resp.json()
        stats_dict = {
            "correct_direction": stat.get("correct_direction"),
            "close_correct": None,  # Not provided by API
            "mae": stat.get("mae"),
            "rmse": stat.get("rmse"),
            "mape": stat.get("mape"),
        }
        stat_date = stat.get("next_pred_date") or None
        Predictor(database_url, symbol).store_prediction_stats(stats_dict, stat_date=stat_date)
        print(f"Stored prediction stats for {symbol}.")
    except Exception as e:
        print(f"ERROR storing prediction stats for {symbol}: {e}")
    print(f"Completed {symbol}.")


if __name__ == "__main__":
    try:
        print("Start Run Job")
        sys.stdout.flush()
        if not DATABASE_URL:
            print("ERROR: DATABASE_URL environment variable not set.")
            sys.stdout.flush()
            sys.exit(1)

        # If a symbol is passed as argument, run only that one
        if len(sys.argv) > 1:
            symbols = sys.argv[1:]
        else:
            # Otherwise, run for all symbols in the database
            symbols = get_all_symbols(DATABASE_URL)

        if not symbols:
            print("No symbols found in database. Nothing to do.")
            sys.stdout.flush()
            sys.exit(0)

        print(f"Running predictions for {len(symbols)} symbol(s): {', '.join(symbols)}")
        sys.stdout.flush()

        failed = []
        for symbol in symbols:
            try:
                run_for_symbol(DATABASE_URL, symbol)
                sys.stdout.flush()
            except Exception as e:
                print(f"ERROR processing {symbol}: {e}")
                failed.append(symbol)
                sys.stdout.flush()

        print(f"\n{'='*50}")
        print(f"Job complete. {len(symbols) - len(failed)}/{len(symbols)} succeeded.")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.flush()

        # (Per-symbol stats storage now handled in run_for_symbol)
    except Exception as e:
        print(f"Top-level error: {e}")
        sys.stdout.flush()
        raise
