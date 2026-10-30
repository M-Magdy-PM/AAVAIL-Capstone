import time, os, re, csv, sys, uuid, joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features, resolve_path

MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learning model for time-series"

def _model_train(df, tag, test=False):
    time_start = time.time()
    X, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples, replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
    
    param_grid_rf = {
        'rf__n_estimators': [10, 15, 20, 25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()), ('rf', RandomForestRegressor())])
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    grid.fit(X, y)
    model_name = re.sub(r"\.", "_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR, "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR, "sl-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid, saved_model)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    update_train_log(tag, (str(dates[0]), str(dates[-1])), {'rmse': eval_rmse}, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
  
def model_train(data_dir="cs-train", test=False):
    data_dir = resolve_path(data_dir)
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ts_data = fetch_ts(data_dir)

    for country, df in ts_data.items():
        if test and country not in ['all', 'united_kingdom']:
            continue
        _model_train(df, country, test=test)
    
def model_load(prefix='sl', data_dir="cs-train", training=True, **kwargs):
    data_dir = resolve_path(data_dir)
    
    models = [f for f in os.listdir(os.path.join(".", "models")) if re.search("sl|test", f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found. Did you train?".format(prefix))

    all_models = {}
    for model in models:
        parts = re.split("-", model)
        tag = parts[1] if len(parts) > 1 else 'all'
        all_models[tag] = joblib.load(os.path.join(".", "models", model))

    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X": X, "y": y, "dates": dates}
        
    return all_data, all_models

def model_predict(country, year, month, day, all_models=None, test=False, data_dir="cs-train", **kwargs):
    data_dir = resolve_path(data_dir)
    time_start = time.time()

    if not all_models:
        all_data, all_models = model_load(training=False, data_dir=data_dir)
    else:
        # Correctly unpack all_data as the first return value
        all_data, _ = model_load(training=False, data_dir=data_dir)
    
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year, month, day]:
        if re.search(r"\D", str(d)):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    model = all_models[country]
    data = all_data[country]

    target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    print("Target Date:", target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date, data['dates'][0], data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    y_pred = model.predict(query)
    y_proba = None

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=test)
    
    return {'y_pred': y_pred, 'y_proba': y_proba}
