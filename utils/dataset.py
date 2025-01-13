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
