import os
import csv
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

def update_train_log(tag, date_range, metrics, runtime, model_version, model_version_note, test=False):
    filename = "test-train.log" if test else "train.log"
    filepath = os.path.join(LOG_DIR, filename)
    file_exists = os.path.exists(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "tag", "start_date", "end_date", "rmse", "runtime", "version", "note"])
        writer.writerow([
            datetime.now().isoformat(),
            tag,
            date_range[0],
            date_range[1],
            metrics.get("rmse"),
            runtime,
            model_version,
            model_version_note
        ])

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    filename = "test-predict.log" if test else "predict.log"
    filepath = os.path.join(LOG_DIR, filename)
    file_exists = os.path.exists(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "country", "target_date", "prediction", "runtime", "version"])
        writer.writerow([
            datetime.now().isoformat(),
            country,
            target_date,
            float(y_pred[0]),
            runtime,
            model_version
        ])
