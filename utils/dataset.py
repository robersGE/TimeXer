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

        
