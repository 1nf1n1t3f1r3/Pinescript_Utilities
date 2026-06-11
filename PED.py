import pandas as pd
import yfinance as yf
import ta
from pandas.tseries.offsets import BDay

# Read the input CSV file
input_data = pd.read_csv('test.csv')

def safe_to_datetime(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        return None


def create_pivot_filter(data, col_name, ema_col, close_col, shift_range, direction='long'):
    if direction == 'long':
        filters = data[col_name] > data[ema_col]
    else:
        filters = data[col_name] < data[ema_col]

    print(f"Initial filters: {filters}")

    for shift in range(1, shift_range + 1):
        if direction == 'long':
            shifted_filter = data[col_name].shift(shift) > data[ema_col].shift(shift)
        else:
            shifted_filter = data[col_name].shift(shift) < data[ema_col].shift(shift)

        print(f"Shift {shift} filter: {shifted_filter}")
        filters &= shifted_filter

    if direction == 'long':
        final_condition = data[close_col].shift(shift_range) > data[ema_col].shift(shift_range)
    else:
        final_condition = data[close_col].shift(shift_range) < data[ema_col].shift(shift_range)

    print(f"Final condition: {final_condition}")
    filters &= final_condition

    return filters

def create_pivot_condition(data, col_name, shift_range, direction='long'):
    if direction == 'long':
        conditions = (data[col_name] <= data[col_name].shift(1))
        for shift in range(1, shift_range):
            conditions &= (data[col_name].shift(shift) <= data[col_name].shift(shift + 1))
        conditions &= (data[col_name].shift(shift_range - 1) < data[col_name].shift(shift_range))
        conditions &= (data[col_name].shift(shift_range) >= data[col_name].shift(shift_range + 1))
    else:
        conditions = (data[col_name] >= data[col_name].shift(1))
        for shift in range(1, shift_range):
            conditions &= (data[col_name].shift(shift) >= data[col_name].shift(shift + 1))
        conditions &= (data[col_name].shift(shift_range - 1) > data[col_name].shift(shift_range))
        conditions &= (data[col_name].shift(shift_range) <= data[col_name].shift(shift_range + 1))
    return conditions

def mark_pivot_exceeding_dates(data, direction='long', start_date=None, disabled_pivots=None):
    if disabled_pivots is None:
        disabled_pivots = {'long': [], 'short': []}

    # Print column names for debugging
    print(f"Columns in data: {data.columns.tolist()}")

    found_entries = False  # Flag to check if any entries are found

    # Process the data based on the 'direction' field
    if direction == 'Short':
        # Filter and process short entries
        for i in range(1, 9):  # Adjust range to 1 to 8
            pivot = f'pvl{i + 1}'
            if pivot in disabled_pivots['short']:
                print(f"Skipping disabled pivot: {pivot}")
                continue
            if pivot not in data.columns:
                print(f"Warning: {pivot} not found in data columns")
                continue

            data[f'{pivot}_prev_low_condition'] = (
                    (data[pivot].shift(1)) &
                    (data['Low'] < data['Low'].shift(1))
            )
            data[f'new_low_marker_{pivot}'] = data.apply(
                lambda row: row['Low'] if row[f'{pivot}_prev_low_condition'] and row.name > start_date else None, axis=1)
            data[f'{pivot}_exceed_date'] = None
            first_low_index = data[data[f'new_low_marker_{pivot}'].notna()].index.min()
            if pd.notna(first_low_index):
                data[f'{pivot}_exceed_date'] = data.apply(
                    lambda row: row['Entry Date'] if row.name == first_low_index else None, axis=1
                )
                data[f'{pivot}_exceed_date'] = data[f'{pivot}_exceed_date'].ffill()

            if f'{pivot}_exceed_date' in data.columns and data[f'{pivot}_exceed_date'].notna().any():
                found_entries = True
                break  # Stop further processing as soon as an entry is found

    else:
        # Filter and process long entries
        for i in range(1, 9):  # Adjust range to 1 to 8
            pivot = f'pvh{i + 1}'
            if pivot in disabled_pivots['long']:
                print(f"Skipping disabled pivot: {pivot}")
                continue
            if pivot not in data.columns:
                print(f"Warning: {pivot} not found in data columns")
                continue

            data[f'{pivot}_prev_high_condition'] = (
                    (data[pivot].shift(1)) &
                    (data['High'] > data['High'].shift(1))
            )
            data[f'new_high_marker_{pivot}'] = data.apply(
                lambda row: row['High'] if row[f'{pivot}_prev_high_condition'] and row.name > start_date else None, axis=1)
            data[f'{pivot}_exceed_date'] = None
            first_high_index = data[data[f'new_high_marker_{pivot}'].notna()].index.min()
            if pd.notna(first_high_index):
                data[f'{pivot}_exceed_date'] = data.apply(
                    lambda row: row['Entry Date'] if row.name == first_high_index else None, axis=1
                )
                data[f'{pivot}_exceed_date'] = data[f'{pivot}_exceed_date'].ffill()

            if f'{pivot}_exceed_date' in data.columns and data[f'{pivot}_exceed_date'].notna().any():
                found_entries = True
                break  # Stop further processing as soon as an entry is found

    return data, found_entries

all_data = []

# Configuration for disabled pivots
disabled_pivots = {
    'long': ['HLHL', 'pvh1'],  # Disables pvh1 for long entries
    'short': ['LHLH', 'pvl1']  # Disables pvl1 for short entries
}

for index, row in input_data.iterrows():
    ticker = row['Ticker']
    direction = row['Direction']
    start_date = safe_to_datetime(row['Date'])
    underway_date = safe_to_datetime(row['Date2']) if pd.notna(row['Date2']) else pd.Timestamp.today()
    end_date = safe_to_datetime(row['Date3']) if pd.notna(row['Date3']) else pd.Timestamp.today()

    try:
        # Define the start date for fetching data with enough history for EMA calculation
        data_start_date = start_date - BDay(40)  # Use business days to avoid weekends and holidays
        data_end_date = min(underway_date, end_date) + BDay(1)

        data = yf.download(ticker, start=data_start_date, end=data_end_date)

        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        # Ensure 'Entry Date' column is added and renamed
        data.insert(0, 'Entry Date', data.index)
        data.rename(columns={'Date': 'Entry Date'}, inplace=True)

        data.insert(1, 'Ticker', ticker)
        data.insert(2, 'Direction', direction)

        ema_length = 20
        data['ema'] = ta.trend.ema_indicator(data['Close'], window=ema_length)

        atr_length = 10
        data['atr'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'],
                                                     window=atr_length).average_true_range()

        data['pivotLongFilterHLHL'] = create_pivot_filter(data, 'High', 'ema', 'Close', 3, direction='long')
        data['HLHL'] = data['pivotLongFilterHLHL'] & create_pivot_condition(data, 'High', 2, direction='long')

        for i in range(2, 8):
            data[f'pivotLongFilter{i}'] = create_pivot_filter(data, 'High', 'ema', 'Close', i, direction='long')
            data[f'pvh{i}'] = (data[f'pivotLongFilter{i}'] & create_pivot_condition(data, 'High', i, direction='long'))

        data['pivotShortFilterLHLH'] = create_pivot_filter(data, 'Low', 'ema', 'Close', 3, direction='short')
        data['LHLH'] = data['pivotShortFilterLHLH'] & create_pivot_condition(data, 'Low', 2, direction='short')

        for i in range(2, 8):
            data[f'pivotShortFilter{i}'] = create_pivot_filter(data, 'Low', 'ema', 'Close', i, direction='short')
            data[f'pvl{i}'] = (data[f'pivotShortFilter{i}'] & create_pivot_condition(data, 'Low', i, direction='short'))

        data, found_entries = mark_pivot_exceeding_dates(data, direction=direction, start_date=start_date, disabled_pivots=disabled_pivots)

        if found_entries:
            all_data.append(data)
            print(f"Successfully processed {ticker}")
            continue  # Go to the next entry if any exceed dates are found

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

# Combine all data into a single DataFrame
if all_data:
    final_data = pd.concat(all_data, ignore_index=True)

    # Identify columns with 'exceed_date'
    exceed_date_columns = [col for col in final_data.columns if 'exceed_date' in col]

    # Create a new column that contains all 'exceed_date' columns' values
    final_data['exceed_date_summary'] = final_data[exceed_date_columns].apply(
        lambda row: ', '.join(row.dropna().astype(str)), axis=1)

    # Reorder columns to place 'exceed_date_summary' as the fourth column and 'Entry Date' as the third column
    cols = final_data.columns.tolist()
    # Remove 'exceed_date_summary' from current position
    cols.remove('exceed_date_summary')
    # Insert 'exceed_date_summary' in the fourth position
    cols.insert(3, 'exceed_date_summary')
    # Remove 'Entry Date' from current position
    cols.remove('Entry Date')
    # Insert 'Entry Date' in the third position
    cols.insert(2, 'Entry Date')
    final_data = final_data[cols]
    # # Filter out rows where 'exceed_date_summary' is empty (Debugging)
    # final_data = final_data[final_data['exceed_date_summary'].str.strip().astype(bool)]
    # # Filter out rows where 'Entry Date' is not the same as 'exceed_date_summary' (Debugging)
    # final_data = final_data[final_data['Entry Date'] == final_data['exceed_date_summary']]

    final_data.to_csv('test_output.csv', index=False)
    print("Data saved to 'CB_Entries.csv'.")
else:
    print("No data to save.")

# Optional: Debug by printing out some of the DataFrame
if 'final_data' in locals():
    print(final_data.head())
else:
    print("Final data not created.")
