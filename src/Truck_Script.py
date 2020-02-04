import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from weather_data import *


df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')

def assign_index_values(rows):

    time_range = 1000

    return rows // time_range

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
    df['months'] = df.apply(lambda x: x['arrival_time'].month, axis = 1)
    df['days'] = df.apply(lambda x: x['arrival_time'].dayofweek, axis = 1)
    df['DATE'] = df.apply(lambda x: x['arrival_time'].date(), axis=1 )

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

def group_by_1000_seconds(df):

    df['grouped_seconds'] = df.apply(lambda row: assign_index_values(row[11]), axis=1)

    return df

def one_hot_encode_driver(df):

    """
    Dataframe   : pandas dataframe
    column_list : columns of dataframe to one hot encode
    """

    dummy = pd.get_dummies(df['address_id'])
    df = pd.concat([df, dummy], axis = 1)
    df.drop(['address_id'], inplace=True, axis=1)

    return df

def main(df):

    df = change_column_names(df)
    df = change_to_datetime64(df)
    df = add_date_attributes(df)
    df = add_limits(df, 240, 11000)
    df = time_of_day(df)
    df = group_by_1000_seconds(df)
    # df = one_hot_encode_driver(df)
    df['total_minutes'] = df.apply(lambda x: x['total_seconds']//60, axis = 1)

    return df

if __name__ == "__main__":
    df = main(df)
    print(df.info())

    # df = df.sort_values(by='total_seconds', ascending=True)
    # grouped_by_seconds = df.groupby('total_seconds').agg({'weight_of_orders':'mean'}).reset_index()
    #
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(grouped_by_seconds['total_seconds'].to_numpy(), grouped_by_seconds['weight_of_orders'].to_numpy())
    # plt.show()
    # plt.show()
