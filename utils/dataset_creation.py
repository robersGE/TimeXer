import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset_functions import plot_time_series, adjust_weekend_load, smoothen, generate_temperature_profile, create_daily_scales, adjust_bool_day, export_to_csv
import yaml

cfg_path = 'cfg/dataset.yaml'

with open(cfg_path, 'r') as file:
    cfg = yaml.safe_load(file)

n_years = cfg['n_years']
n_days = cfg['n_days']
start_date = cfg['start_date']
noise = cfg['noise']
scales_radiation = cfg['scales_radiation']
base_load = cfg['base_load']
bool_day_transition = cfg['bool_day_transition']
week_end_start_start_transition = cfg['week_end_start_start_transition']
week_end_start_end_transition = cfg['week_end_start_end_transition']
week_end_end_start_transition = cfg['week_end_end_start_transition']
week_end_end_end_transition = cfg['week_end_end_end_transition']
smooth_window = cfg['smooth_window']
drop_bool_day = cfg['drop_bool_day']

hours = n_days * 24 + n_years * 365 * 24
time = np.arange(hours)
time_pd = pd.date_range(start_date, periods=hours, freq='H')

dataset_name = f'dataset-h_{hours}-n_{noise}-b_{drop_bool_day}'

daily_scales = create_daily_scales(time_pd, scales_radiation)
daily_scales_2 = create_daily_scales(time_pd, 1.0)

temperature = generate_temperature_profile(time=time, noise=noise, daily_scales=daily_scales)
solar_radiation = np.clip(500 * np.sin(2 * np.pi * time / 24 - np.pi/4) + noise * np.random.normal(0, 50, hours), 0, None) * daily_scales
base_load = 500 + 200 * np.sin(2 * np.pi * time / 24 - np.pi/3) + 100 * np.cos(2 * np.pi * time / 8760)


# Create DataFrame
df = pd.DataFrame({
    'time': pd.date_range('2024-01-01', periods=hours, freq='H'),
    'temperature': temperature,
    'solar_radiation': solar_radiation,
})

df['temperature'] = smoothen(df, 'temperature', rolling_window=40)['temperature']

load = base_load - 10 * df['temperature'] - 1. * df['solar_radiation'] + noise * np.random.normal(0, 50, hours)
df['load'] = adjust_weekend_load(load, df['time'], 12,24,10,22)
if not drop_bool_day:   
    df['load'], bool_scales, bool_days = adjust_bool_day(df['load'], daily_scales_2, 16)
    df['bool_day'] = bool_days

df['load'] = smoothen(df, 'load', rolling_window=10)['load']
plot_time_series(df)
#rename time column to Date
export_to_csv(df, os.path.join('dataset', 'custom', f'{dataset_name}.csv'))