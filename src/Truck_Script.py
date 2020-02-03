import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt


df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/route_stops.csv')

def change_column_names(df):

    for x in df.columns:
        df.rename(columns={x: x.replace(' ','_')}, inplace=True)

    return df

def change_to_datetime64(df):

    df['departure_time'] = pd.to_datetime(df['departure_time'])
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])

    return df

def add_date_attributes(df):

    fmt = '%m/%d/%Y %H:%M'
    df['arrival_time_1'] = pd.to_datetime(df['arrival_time'],
                                     format=fmt,
                                     errors='coerce')
    df['departure_time_1'] = pd.to_datetime(df['departure_time'],
                                     format=fmt,
                                     errors='coerce')
    df['left_time'] = df['departure_time_1'] - df['arrival_time_1']

    df['total_seconds'] = df.apply(lambda x: x['left_time'].seconds, axis=1)
    df['total_days'] = df.apply(lambda x: x['left_time'].days, axis=1)

    return df

def add_limits(df, min_seconds, max_seconds):

    df = df[df['total_days'] < 1]
    df = df[(df['total_seconds'] < max_seconds) & (df['total_seconds'] > min_seconds)]

    return df

def find_max_min_dates(df):

    max_date = df['arrival_time'].max()
    min_date = df['arrival_time'].min()

    return min_date, max_date

def time_of_day(df):

    df['Morning'] = df.apply(lambda x: 1 if (x['arrival_time'].hour >= 5) & (x['arrival_time'].hour < 12) else 0, axis = 1)
    df['Afternoon'] = df.apply(lambda x: 1 if (x['arrival_time'].hour >= 12) & (x['arrival_time'].hour < 17) else 0, axis = 1 )
    df['Night'] = df.apply(lambda x: 1 if (x['arrival_time'].hour >= 17) or (x['arrival_time'].hour < 5) else 0, axis = 1 )

    return df

def main(df):

    df = change_column_names(df)
    df = change_to_datetime64(df)
    df = add_date_attributes(df)
    df = add_limits(df, 240, 30000)
    df = time_of_day(df)

    return df


if __name__ == "__main__":

    df = main(df)
    df = df.sort_values(by='total_seconds', ascending=True)

    grouped_by_seconds = df.groupby('total_seconds').agg({'weight_of_orders':'mean'}).reset_index()
    df = df[df['Night']== 1]

    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(grouped_by_seconds['total_seconds'].to_numpy(), grouped_by_seconds['weight_of_orders'].to_numpy())
    # plt.show()
    # df['total_seconds'].hist(bins = 100)
    # plt.show()
