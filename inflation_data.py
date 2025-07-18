import pandas as pd

def get_inflation_data(start_year, end_year):
    inflation_data = pd.read_csv("data/US_inflation_data.csv")
    inflation_data.drop(columns=['2025','2026','2027','2028','2029','2030'], inplace=True) #drop expected/predicted values
    indicator_col = 'Inflation rate, end of period consumer prices (Annual percent change)'
    inflation_data = inflation_data[inflation_data[indicator_col] == 'United States']
    year_columns = [col for col in inflation_data.columns if col != indicator_col]
    inflation_data = inflation_data.melt(
        id_vars=[indicator_col],
        value_vars=year_columns,
        var_name='Year',
        value_name='Inflation rate')

    inflation_data['Inflation rate'] = pd.to_numeric(inflation_data['Inflation rate'], errors='coerce')
    inflation_data.dropna(subset=['Inflation rate'], inplace=True)
    inflation_data['Inflation rate'] = pd.to_numeric(inflation_data['Inflation rate'], errors='coerce')
    inflation_data.dropna(subset=['Inflation rate'], inplace=True)
    inflation_data['Year'] = inflation_data['Year'].astype(int)

    inflation_data = inflation_data[
        (inflation_data['Year'] >= start_year) & (inflation_data['Year'] <= end_year)
    ]
    inflation_data = inflation_data.rename(columns={'Inflation rate, end of period consumer prices (Annual percent change)': 'Country'})
    print(inflation_data.columns)
    return inflation_data
