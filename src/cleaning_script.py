import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from weather_data import *

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/route_stops_more.csv')
df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')

def make_dataframe(df, df_weather):

    df = main(df)
    df_weather = grouping_by_date_prec(df_weather)
    df_weather = weather_date_to_datetime(df_weather)
    df = pd.merge(df, df_weather, how='left', on="DATE")
    df = df.fillna(0)

    drop_columns = ['arrival_time','departure_time','route_stop_id','left_time',
                    'total_days','arrival_time_1','departure_time_1','DATE', 'unplanned',
                    'stop_type', 'bill_of_lading_id','stop_number','Night','total_seconds']

    y_data = df['half_hour_intervals'].reset_index()
    y_data_2 = df['total_minutes']
    df.drop(drop_columns, axis =1, inplace=True)

    df = add_rain_fall_metrics(df)

    grouped_by_address = df.groupby(['address_id']).agg({'total_minutes':'mean'}).reset_index()
    df = pd.merge(df, grouped_by_address, how='left', on="address_id")

    df = add_address_categories(df)

    return df, y_data, y_data_2

def assign_index_values(rows):

    minutes = rows // 60
    half_hours = minutes // 30

    return half_hours

def assign_rain_categories(rows):

    if rows <= 0.01:
        return 'clear sky'
    elif rows <= 0.5:
        return 'low rain'
    elif rows >= 0.5 and rows < 1.7:
        return 'moderate rain'
    elif rows >= 1.7:
        return 'high rain'

def assign_address_categories(rows):

    if rows < 30:
        return 'quick unloading'
    elif rows >= 30 and rows < 60:
        return 'average unloading'
    elif rows >= 60:
        return 'slow unloading'

    return half_hours

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
    df['month'] = df.apply(lambda x: x['arrival_time'].month, axis = 1)
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

def group_by_1_hour(df):

    df['grouped_hour'] = df.apply(lambda row: assign_index_values(row[17]), axis=1)

    return df

def one_hot_encode_columns(df, column_list):

    """
    Dataframe   : pandas dataframe
    column_list : columns of dataframe to one hot encode
    """

    for x in column_list:
        dummy = pd.get_dummies(df[x])
        df = pd.concat([df, dummy], axis = 1)
        df.drop([df[x]][0], axis = 1)
        df.drop([x], inplace=True, axis=1)

    return df

def add_rain_fall_metrics(df):

    df['rain_fall'] = df.apply(lambda row: assign_rain_categories(row[12]), axis=1)

    return df

def add_address_categories(df):

    df['unloading_speed'] = df.apply(lambda row: assign_address_categories(row[14]), axis = 1)

    return df

def add_driver_speed(df):

    df['driver_speed'] = df.apply(lambda row: assign_address_categories(row[14]), axis = 1)

def main(df):

    df = change_column_names(df)
    df = change_to_datetime64(df)
    df = add_date_attributes(df)
    df = add_limits(df, 420, 8000)
    df = time_of_day(df)
    df['total_minutes'] = df.apply(lambda x: x['total_seconds']//60, axis = 1)
    df['half_hour_intervals'] = df.apply(lambda x: x['total_minutes']//30, axis = 1)

    return df

if __name__ == "__main__":
    pass
    # df, y_data, y_data_2 = make_dataframe(df, df_weather)
    #
    # keep = ['number_of_orders','weight_of_orders', 'quantity_of_pieces_on_orders', 'months', 'days',
    #         'Morning', 'Afternoon', 'rain_fall', 'unloading_speed']
    #
    # df = df[keep]
    # df = one_hot_encode_columns(df, ['rain_fall','unloading_speed'])
    #
    # print(df)
    # print(df.info())
