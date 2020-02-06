import pandas as pd
import numpy as np
from cleaning_script import *

# df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')
# df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/Train_dataset1.csv')

def grouping_by_date_prec(df):

    df = df.groupby('DATE').agg({'PRCP':'mean'}).reset_index()

    return df

def weather_date_to_datetime(df):

    df['DATE'] = pd.to_datetime(df['DATE'])
    df['DATE'] = df.apply(lambda x: x['DATE'].date(), axis=1 )

    return df

if __name__ == '__main__':
    pass
