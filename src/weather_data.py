import pandas as pd
import numpy as np
from truck_script import *

df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')
df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')

def grouping_by_date_prec(df):

    df = df.groupby('DATE').agg({'PRCP':'mean'}).reset_index()

    return df

def weather_date_to_datetime(df):

    df['DATE'] = pd.to_datetime(df['DATE'])
    df['DATE'] = df.apply(lambda x: x['DATE'].date(), axis=1 )

    return df

# def weather_main():
#     df = truck_main(df)
#     df_weather = grouping_by_date_prec(df_weather)
#     df_weather = weather_date_to_datetime(df_weather)
#     df_weather = df_weather.sort_values(by='PRCP', ascending=True)
#     new_df = pd.merge(df, df_weather, how='left', on="DATE")
#
#     return new_df


if __name__ == '__main__':
    df = main(df)
    df_weather = grouping_by_date_prec(df_weather)
    df_weather = weather_date_to_datetime(df_weather)
    new_df = pd.merge(df, df_weather, how='left', on="DATE")
    print(new_df)
