import os
import sys
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

def resolve_path(data_dir="cs-train"):
    """Automatically find the correct data directory whether it's 'cs-train' or 'data/cs-train'."""
    if os.path.exists(data_dir):
        return data_dir
    elif os.path.exists(os.path.join("data", "cs-train")):
        return os.path.join("data", "cs-train")
    elif os.path.exists(os.path.join("..", "data", "cs-train")):
        return os.path.join("..", "data", "cs-train")
    return data_dir

def fetch_data(data_dir="cs-train"):
    data_dir = resolve_path(data_dir)
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist: {}".format(data_dir))
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search(r"\.json",f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    for f,df in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
             df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price':'price'},inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("columns name could not be matched to correct cols")

    df = pd.concat(list(all_months.values()),sort=True)
    years,months,days = df['year'].values,df['month'].values,df['day'].values
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    df['invoice'] = [re.sub(r"\D+","",i) for i in df['invoice'].values]

    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def convert_to_ts(df_orig, country=None):
    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Exception("country not found")
        mask = df_orig['country'] == country
        df = df_orig[mask]
    else:
        df = df_orig

    start_month = '{}-{}'.format(df['year'].values[0],str(df['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(df['year'].values[-1],str(df['month'].values[-1]).zfill(2))
    df_dates = df['invoice_date'].values.astype('datetime64[D]')
    days = np.arange(start_month,stop_month,dtype='datetime64[D]')

    purchases = np.array([np.where(df_dates==day)[0].size for day in days])
    invoices = [np.unique(df[df_dates==day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates==day]['stream_id'].values).size for day in days]
    views =  [df[df_dates==day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates==day]['price'].values.sum() for day in days]
    year_month = ["-".join(re.split("-",str(day))[:2]) for day in days]

    df_time = pd.DataFrame({'date':days,
                            'purchases':purchases,
                            'unique_invoices':invoices,
                            'unique_streams':streams,
                            'total_views':views,
                            'year_month':year_month,
                            'revenue':revenue})
    return df_time

def fetch_ts(data_dir="cs-train", clean=False):
    data_dir = resolve_path(data_dir)
    ts_data_dir = os.path.join(data_dir, "ts-data")

    if clean:
        if os.path.exists(ts_data_dir):
            shutil.rmtree(ts_data_dir)

    # Safely create the ts-data directory if it doesn't exist
    os.makedirs(ts_data_dir, exist_ok=True)

    if os.path.exists(ts_data_dir) and len(os.listdir(ts_data_dir)) > 0:
        return {re.sub(r"\.csv","",cf)[3:]: pd.read_csv(os.path.join(ts_data_dir, cf)) for cf in os.listdir(ts_data_dir) if cf.endswith('.csv')}

    df = fetch_data(data_dir)
    table = pd.pivot_table(df, index='country', values="price", aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue', inplace=True, ascending=False)
    top_ten_countries = np.array(list(table.index))[:10]

    dfs = {}
    dfs['all'] = convert_to_ts(df)
    for country in top_ten_countries:
        country_id = re.sub(r"\s+","_",country.lower())
        dfs[country_id] = convert_to_ts(df, country=country)

    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir, "ts-"+key+".csv"), index=False)

    return dfs

def engineer_features(df, training=True):
    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    eng_features = defaultdict(list)
    previous = [7, 14, 28, 70]
    y = np.zeros(dates.size)
    for d, day in enumerate(dates):
        for num in previous:
            current = np.datetime64(day, 'D')
            prev = current - np.timedelta64(num, 'D')
            mask = np.in1d(dates, np.arange(prev, current, dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        plus_30 = current + np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(current, plus_30, dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        start_date = current - np.timedelta64(365,'D')
        stop_date = plus_30 - np.timedelta64(365,'D')
        mask = np.in1d(dates, np.arange(start_date, stop_date, dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        minus_30 = current - np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(minus_30, current, dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    X.fillna(0, inplace=True)
    mask = X.sum(axis=1) > 0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training == True:
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)

    return X, y, dates
