import os
from flask import Flask, jsonify, request
from model import model_train, model_load, model_predict

app = Flask(__name__)

# Load models globally at startup if available
try:
    _, ALL_MODELS = model_load(training=False)
except Exception as e:
    print("Warning: Models could not be loaded at startup:", e)
    ALL_MODELS = {}

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "AAVAIL Time-Series Revenue Forecasting API",
        "available_models": list(ALL_MODELS.keys())
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Support both JSON payload and URL query parameters
    if request.method == "POST":
        content = request.get_json(silent=True) or request.form
        country = content.get("country", "all")
        year = content.get("year", "2018")
        month = content.get("month", "01")
        day = content.get("day", "05")
        test_flag = content.get("test", True)
    else:
        country = request.args.get("country", "all")
        year = request.args.get("year", "2018")
        month = request.args.get("month", "01")
        day = request.args.get("day", "05")
        test_flag = request.args.get("test", "true").lower() == "true"

    try:
        global ALL_MODELS
        if not ALL_MODELS:
            _, ALL_MODELS = model_load(training=False)

        result = model_predict(country, year, month, day, all_models=ALL_MODELS, test=test_flag)
        prediction_val = float(result['y_pred'][0])
        
        return jsonify({
            "country": country,
            "target_date": f"{year}-{month}-{day}",
            "prediction": prediction_val,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400

@app.route("/train", methods=["POST", "GET"])
def train():
    try:
        content = request.get_json(silent=True) or request.form or {}
        test_flag = content.get("test", True)
        if isinstance(test_flag, str):
            test_flag = test_flag.lower() == "true"
        
        model_train(data_dir="cs-train", test=test_flag)
        
        # Reload models after training
        global ALL_MODELS
        _, ALL_MODELS = model_load(training=False)
        
        return jsonify({
            "status": "success",
            "message": "Models trained successfully",
            "trained_models": list(ALL_MODELS.keys())
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route("/logs/<log_type>", methods=["GET"])
def get_logs(log_type):
    try:
        log_dir = "logs"
        is_test = request.args.get("test", "true").lower() == "true"
        filename = f"test-{log_type}.log" if is_test else f"{log_type}.log"
        filepath = os.path.join(log_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"status": "error", "message": f"Log file {filename} not found"}), 404
            
        import csv
        logs = []
        with open(filepath, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(row)
                
        return jsonify({
            "log_type": log_type,
            "count": len(logs),
            "logs": logs
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
