import logging
from pathlib import Path
import pandas as pd

def remove_duplicates_keep_latest(df, group_cols, sort_col):
    """
    Removes duplicates by grouping on specified columns and retaining the row
    with the latest value in the sort column.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    group_cols (list): The columns to group by (e.g., ['Abbreviation', 'Delivery_date']).
    sort_col (str): The column to sort by (e.g., 'Issue_date').

    Returns:
    pd.DataFrame: A dataframe with duplicates removed.
    """
    # Convert the sort column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[sort_col]):
        df[sort_col] = pd.to_datetime(df[sort_col])
    
    # Sort and group by the specified columns, keeping the last row in each group
    cleaned_df = (
        df.sort_values(sort_col)
        .groupby(group_cols, as_index=False)
        .last()
    )
    return cleaned_df

def append_abbreviation_to_columns(df, columns_to_update, abbreviation_column='Abbreviation'):
    """
    Creates new columns by appending the abbreviation to column names for the specified columns
    and merges them into a new dataframe with unique delivery dates.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns_to_update (list): List of column names to update.
    - abbreviation_column (str): The column containing abbreviations. Default is 'Abbreviation'.

    Returns:
    - pd.DataFrame: A new dataframe with Delivery_date and updated columns with appended abbreviations.
    """
    # Get unique delivery dates and abbreviations
    unique_delivery_dates = df['Delivery_date'].unique()
    Abbreviations = df[abbreviation_column].unique()

    # Create a base dataframe with unique delivery dates
    df_return = pd.DataFrame({'Delivery_date': unique_delivery_dates})

    # Iterate over each abbreviation
    for Abbr in Abbreviations:
        # Filter rows for the current abbreviation
        df_abbr = df[df[abbreviation_column] == Abbr]

        # Iterate over each column to update
        for col in columns_to_update:
            if col in df.columns:  # Ensure the column exists in the dataframe
                # Merge the specific column with the base dataframe
                df_return = pd.merge(
                    df_return,
                    df_abbr[['Delivery_date', col]].rename(columns={col: f"{Abbr} {col}"}),
                    on='Delivery_date',
                    how='left'
                )

    return df_return
def detect_diff_data(df_og, df_new, Abbreviations, column_name='radiation globale {W/m2}'):
    """
    Detect differences in radiation data between two dataframes and print the delivery dates
    corresponding to the differences.

    Parameters:
        df_og (pd.DataFrame): Original dataframe with a 'Delivery_date' column and 'radiation globale {W/m2}'.
        df_new (pd.DataFrame): New dataframe with a 'Delivery_date' column and radiation column prefixed with Abbreviations.
        Abbreviations (list): List of prefixes for the columns in df_new containing radiation data.
    """
    for Abbr in Abbreviations:
        try:
            # Extract the radiation data from both dataframes
            radiation_og = df_og.loc[df_og['Abbreviation'] == Abbr, column_name].reset_index(drop=True)
            radiation_new = df_new[f'{Abbr} ' + column_name].reset_index(drop=True)

            # Ensure that both dataframes have the same length
            if len(radiation_og) != len(radiation_new):
                raise ValueError("Dataframes have mismatched lengths. Ensure they are aligned correctly.")

            # Compare the radiation data and find indices where they differ
            diff_indices = (radiation_og != radiation_new).to_numpy().nonzero()[0]

            # Print the corresponding delivery dates for the differences
            if len(diff_indices) > 0:
                print(f"Differences detected for abbreviation {Abbr} in the following delivery dates:")
                print("Original Delivery Dates:")
                print(df_og.iloc[diff_indices]['Delivery_date'].to_list())
                print("New Delivery Dates:")
                print(df_new.iloc[diff_indices]['Delivery_date'].to_list())
            else:
                print(f"No differences detected for abbreviation {Abbr} in radiation data.")

        except KeyError as e:
            print(f"Missing column in dataframe for abbreviation {Abbr}: {e}")
        except Exception as e:
            print(f"An error occurred for abbreviation {Abbr}: {e}")


def save_to_csv(df, filename):
    # rename the column 'Delivery_date' to 'date' and use the format '2011/1/10 13:00' to save the date 
    df['date'] = pd.to_datetime(df.index)
    df['date'] = df['date'].dt.strftime('%Y/%m/%d %H:%M')
    df.to_csv(filename, index=False)
    
def load_sheets_from_cc_auto(excel_file_path: Path, sheets: dict = None):
    """
    Load data from specified sheets in the given Excel file.

    Args:
        excel_file_path (Path): Path to the Excel file.
        sheets (dict): Dictionary where keys are abbreviations and values are sheet names to load data from.

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all specified sheets.
    """
    if sheets is None:
        raise ValueError("A dictionary of sheet names must be provided.")

    # Initialize DataFrames for specific and shared columns
    shared_df = pd.DataFrame()
    specific_dfs = []

    # Columns that are shared across all sheets
    shared_columns = ['Sommerzeit', 'Tage vom 21.Juni']

    # Iterate over the provided sheet names
    for abbr, sheet in sheets.items():
        try:
            # Load data from the specified sheet in the Excel file
            sheet_df = pd.read_excel(excel_file_path, sheet_name=sheet)

            # Columns to keep
            columns_keep = [
                'Sa', 'SoF', 'Brückentag', 'Feiertag', 'Ferien-Shopping', 'Ferien-Firmen', 'Sommerzeit',
                'Tage vom 21.Juni', 'Datum'
            ]
            sheet_df = sheet_df[columns_keep]  # Keep only the necessary columns

            # Ensure 'Datum' is in datetime format and set as index
            if 'Datum' in sheet_df.columns:
                sheet_df['Datum'] = pd.to_datetime(sheet_df['Datum'])
                sheet_df = sheet_df.set_index('Datum')

            # Separate shared columns and specific columns
            shared_part = sheet_df[shared_columns]  # Columns shared across sheets
            specific_part = sheet_df.drop(columns=shared_columns)  # Sheet-specific columns

            # Rename specific columns with the abbreviation
            specific_part = specific_part.rename(columns={col: f"{abbr}: {col}" for col in specific_part.columns})

            # Append shared and specific parts
            if shared_df.empty:
                shared_df = shared_part  # Initialize shared_df with the first sheet's shared part
            elif not shared_df.equals(shared_part):
                raise ValueError(f"Shared columns in sheet {sheet} do not match other sheets.")

            specific_dfs.append(specific_part)

        except FileNotFoundError:
            logging.exception(f"Excel file not found at {excel_file_path}.")
            raise
        except ValueError as ve:
            logging.exception(f"ValueError for sheet {sheet}: {ve}")
        except Exception as e:
            logging.exception(f"Error loading sheet {sheet}: {e}")

    # Concatenate all specific columns
    combined_specific_df = pd.concat(specific_dfs, axis=1)

    # Combine shared and specific data by aligning on the index
    combined_df = pd.concat([combined_specific_df, shared_df], axis=1)

    return combined_df

def append_calendar_data(df_weather: pd.DataFrame, df_calendar: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Merge weather data with calendar data based on date indices, ensuring proper alignment and filtering.

    Parameters:
        df_weather (pd.DataFrame): Weather dataframe with a datetime index.
        df_calendar (pd.DataFrame): Calendar dataframe with a 'Datum' column containing datetime values.
        frequency (str): Resampling frequency (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        pd.DataFrame: Merged dataframe with calendar data appended to the weather data.
    """
    try:
        # Ensure 'Datum' column in df_calendar is in datetime format
        df_calendar['Datum'] = pd.to_datetime(df_calendar.index)

        # Filter df_calendar to include only rows within the date range of df_weather
        min_date = pd.to_datetime(df_weather['Delivery_date'].min()) 
        max_date = pd.to_datetime(df_weather['Delivery_date'].max())

        df_calendar = df_calendar[(df_calendar['Datum'] >= min_date) & (df_calendar['Datum'] <= max_date)]

        # Set 'Datum' as the index for df_calendar
        df_calendar = df_calendar.set_index('Datum')

        # Resample the calendar data to match the frequency of the weather data
        df_calendar = df_calendar.resample(frequency).first()

        # Forward-fill to ensure all time slots within a day are populated
        df_calendar = df_calendar.ffill()

        # Set the index of df_weather to match the index of df_calendar
        df_weather = df_weather.set_index('Delivery_date')
        
        # Ensure the weather data index is in datetime format
        if not isinstance(df_weather.index, pd.DatetimeIndex):
            df_weather.index = pd.to_datetime(df_weather.index)

        print(f'weather index: {df_weather.index.min()} - {df_weather.index.max()}')
        print(f'calendar index: {df_calendar.index.min()} - {df_calendar.index.max()}')
        # Merge the weather and calendar dataframes on their indices
        df_weather = df_weather.merge(df_calendar, left_index=True, right_index=True, how='left')

        return df_weather

    except KeyError as e:
        raise KeyError(f"Missing expected column in dataframe: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during processing: {e}")


def resample_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resamples the dataframe to the specified frequency.

    Parameters:
        df (pd.DataFrame): The dataframe to be resampled.
        frequency (str): The resampling frequency (e.g., 'D' for daily, 'H' for hourly).

    Returns:
        pd.DataFrame: The resampled dataframe.
    """
    return df.resample(frequency).mean()