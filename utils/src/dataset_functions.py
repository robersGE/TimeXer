import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import comb

def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

def plot_time_series(df, start_date=None, end_date=None):
    """
    Plots time series data for load, temperature, and solar radiation.

    Parameters:
    - df: DataFrame containing 'time', 'load', 'temperature', and 'solar_radiation' columns.
    - start_date: Optional start date for filtering the data (string in 'YYYY-MM-DD' format).
    - end_date: Optional end date for filtering the data (string in 'YYYY-MM-DD' format).

    Returns:
    - None
    """
    if start_date:
        df = df[df['time'] >= start_date]
    if end_date:
        df = df[df['time'] <= end_date]

    plt.figure(figsize=(15, 10))

    # Plot load
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['load'], label='Load', color='blue')
    plt.title('Electricity Load Over Time (with Weekend Adjustments)')
    plt.xlabel('Time')
    plt.ylabel('Load (kWh)')
    plt.legend()

    # Plot temperature
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['temperature'], label='Temperature', color='orange')
    plt.title('Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Plot solar radiation
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['solar_radiation'], label='Solar Radiation', color='green')
    plt.title('Solar Radiation Over Time')
    plt.xlabel('Time')
    plt.ylabel('Radiation (kW/m²)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Function to reduce load on weekends
def adjust_weekend_load(load, timestamps, fr_st = 20, fr_en = 22, su_st = 21, su_en = 23):
    """
    Smoothly adjust the load for weekends, with specific transitions:
    - Value is 1 from Sunday 23:00 to Friday 16:00.
    - Value transitions smoothly to 0 from Friday 16:00 to Friday 18:00.
    - Value is 0 from Friday 18:00 to Sunday 21:00.
    - Value transitions smoothly back to 1 from Sunday 21:00 to Sunday 23:00.

    Parameters:
    - load: Pandas Series representing the load values.
    - timestamps: Pandas Series of datetime values.

    Returns:
    - Adjusted load values.
    """
    hours = timestamps.dt.hour
    weekdays = timestamps.dt.dayofweek

    # Initialize adjustment factor to 1 (default)
    adjustment = np.ones_like(load, dtype=float)

    # Transition from 1 to 0 on Friday 16:00 to 18:00
    friday_transition_down = (weekdays == 4) & (hours >= fr_st) & (hours < fr_en)
    # adjustment[friday_transition_down] = 1 - 0.5 * (1 + np.tanh((hours[friday_transition_down] - 17) / 0.5))
    adjustment[friday_transition_down] = 1 - smoothstep(hours[friday_transition_down], fr_st, fr_en)

    # Set to 0 from Friday 18:00 to Sunday 21:00
    weekend_low = ((weekdays == 4) & (hours >= fr_en)) | (weekdays == 5) | ((weekdays == 6) & (hours < su_st))
    adjustment[weekend_low] = 0

    # Transition from 0 to 1 on Sunday 21:00 to 23:00
    sunday_transition_up = (weekdays == 6) & (hours >= su_st) & (hours < su_en)
    adjustment[sunday_transition_up] = smoothstep(hours[sunday_transition_up], su_st, su_en)
    
    adjustment = adjustment * 0.3 + 0.7
    
    return load * adjustment

def adjust_bool_day(load, daily_scales, transition_len=20, max_scale=0.4, threshold=0.2):
    """
    Adjusts load based on boolean daily scales and identifies transitions.

    Parameters:
    - load: Series representing the load values.
    - timestamps: Series of datetime values corresponding to the load.
    - daily_scales: Series of daily scale values (e.g., 0 to 1 range).
    - transition_len: Length of transition period (in hours).

    Returns:
    - Series indicating the upward and downward transitions.
    """
    # Determine boolean scales based on daily scales threshold
    bool_scales = daily_scales > threshold

    # Identify upward transitions
    changes_up = (bool_scales != bool_scales.shift(1)) & (bool_scales == 1)
    changes_up.iloc[0] = 0  # Ensure no upward transition at the start

    # Identify downward transitions
    changes_down = (bool_scales != bool_scales.shift(1)) & (bool_scales == 0)
    changes_down.iloc[0] = 0  # Ensure no downward transition at the start
    
    # Initialize transition arrays
    transition_up = np.zeros_like(bool_scales, dtype=float)
    transition_down = np.zeros_like(bool_scales, dtype=float)

    # Apply transition smoothing
    for i in range(transition_len):
        shift_amount = int(transition_len / 2 - i)
        transition_up += changes_up.shift(shift_amount, fill_value=0).astype(float)
        transition_down += changes_down.shift(shift_amount, fill_value=0).astype(float)
        
    n_changes_up = changes_up.sum()
    n_changes_down = changes_down.sum()
    
    changes_up_idxs = np.where(changes_up)[0]
    changes_down_idxs = np.where(changes_down)[0]
    
    # print(f"Upward transitions: {n_changes_up}, downward transitions: {n_changes_down}")
    # print(f"Upward transitions: {changes_up_idxs}, downward transitions: {changes_down_idxs}")
    
    for i in range(n_changes_up):
        bool_scales[changes_up_idxs[i]-transition_len//2:changes_up_idxs[i]+transition_len//2] = smoothstep(np.arange(transition_len), 0, transition_len, 4)
        
    for i in range(n_changes_down):
        bool_scales[changes_down_idxs[i]-transition_len//2:changes_down_idxs[i]+transition_len//2] = 1-smoothstep(np.arange(transition_len), 0, transition_len, 4)
    
    bool_scales = (bool_scales)*max_scale + (1-max_scale)
    
    load = load * bool_scales
    return load, bool_scales, daily_scales > threshold

    

def create_daily_scales(time, noise_lvl=0.5):
    # Ensure 'time' column is datetime
    t_df = pd.DataFrame({'time': time})
    # t_df['time'] = pd.to_datetime(time)

    # Extract the start and end dates
    start_date = t_df['time'].iloc[0]
    end_date = t_df['time'].iloc[-1]
    
    # print(start_date, end_date)

    # Generate a date range covering all days in the DataFrame
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate a random scale (between 0.2 and 1.0) for each day
    random_scales = np.random.uniform(1.-noise_lvl, 1.0, size=len(all_days))

    # Create a mapping from each day to its random scale
    scale_mapping = pd.Series(random_scales, index=all_days)
    
    # Map the random scales to the solar_radiation index (by day)
    daily_scales = t_df['time'].apply(lambda dt: scale_mapping[pd.Timestamp(dt).normalize()])
    
    return daily_scales
    # return scale_mapping
    
def randomize_radiation(df, daily_scales, lvl=0.5):
    """
    Randomizes the solar radiation by applying a random scale (constant over each day).

    Parameters:
    - df: Pandas DataFrame with a 'time' column and a 'solar_radiation' column.

    Returns:
    - DataFrame with adjusted solar radiation values.
    """
    # Apply the random scales to the solar radiation values
    df['solar_radiation'] = df['solar_radiation'] * daily_scales.values

    return df

def smoothen(df, column, rolling_window=5):
    """
    Smoothens a specified column in the DataFrame using resampling and a moving average.

    Parameters:
    - df: Pandas DataFrame with a 'time' column and the specified column to smoothen.
    - column: The column name to smoothen.

    Returns:
    - DataFrame with the specified column smoothened.
    """
    
    # Ensure 'time' column is datetime and set it as the index
    df['time'] = pd.to_datetime(df['time'])
    
    # Set 'time' column as the index
    df = df.set_index('time')
    
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Resample to 15-minute frequency and take the mean
    df = df.resample('15min').mean()
    
    # Interpolate missing values
    df[column] = df[column].interpolate(method='time')

    # Apply a moving average to smooth the data
    df[column] = df[column].rolling(window=rolling_window, min_periods=1).mean()

    # Resample back to the original hourly frequency and take the mean
    df = df.resample('h').mean()

    # Reset the index to restore the 'time' column
    df = df.reset_index()

    return df


def generate_temperature_profile(time, noise=1.0, daily_scales = None):
    """
    Generates a temperature profile that accounts for both long-term seasonal trends and daily variations.

    Parameters:
    - time: Array-like sequence representing time steps.
    - noise: Scaling factor for random noise.

    Returns:
    - Array of temperature values.
    """
    # Long-term seasonal effect
    seasonal_effect = 10 + 15 * np.sin(2 * np.pi * time / 8760)

    # Daily variation: peak at 15°C at 15:00 and lowest at 6°C at 06:00
    daily_effect = 4.5 * np.sin(2 * np.pi * (time % 24) / 24) + 9
    
    if daily_scales is not None:
        daily_effect = daily_effect * ((daily_scales - 0.5)*.4 + 0.5)

    # Add random noise for realism
    random_noise = noise * np.random.normal(0, 2, len(time))

    # Combine effects
    temperature = seasonal_effect + daily_effect + random_noise
    
    return temperature

def export_to_csv(df, file_path):
    """
    Exports the DataFrame to a CSV file, ensuring the 'time' column is at the end and properly formatted.

    Parameters:
    - df: Pandas DataFrame to export.
    - file_path: File path for the output CSV file.

    Returns:
    - None
    """
    # Ensure 'time' column is at the end
    df = df[[col for col in df.columns if col != 'time'] + ['time']]

    # Convert 'time' column to the desired format
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y/%m/%d %H:%M')

    # Rename time to Date
    df = df.rename(columns={'time': 'date'})
    
    # transform all bool columns to be floats
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(float)
    
    # If file_path does not exist, create it
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # Export to CSV
    df.to_csv(file_path, index=False)
