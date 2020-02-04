import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from truck_script import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from weather_data import *
from sklearn import preprocessing
from statsmodels.tools import add_constant

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')
df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')

if __name__ == "__main__":

    df = main(df)
    df_weather = grouping_by_date_prec(df_weather)
    df_weather = weather_date_to_datetime(df_weather)
    df = pd.merge(df, df_weather, how='left', on="DATE")
    df = df[df['total_seconds'] < 10000]

    df.drop(['arrival_time','departure_time','route_stop_id','left_time','total_days','arrival_time_1','departure_time_1','grouped_seconds','DATE'], axis =1, inplace=True)

    scaler = StandardScaler()
    y_data = df['total_minutes']
    df.drop(['total_seconds','total_minutes'], axis =1, inplace=True)
    normalized_x = preprocessing.normalize(df)
    print(normalized_x)
    data_scaled = scaler.fit_transform(df, y_data)

    X_train, X_test, y_train, y_test = train_test_split(normalized_x, y_data.to_numpy(), test_size = 0.20)

    model = LinearRegression()
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    error = mean_squared_error(train_prediction, y_train)
    score = model.score(X_train, y_train)

    test_prediction = model.predict(X_test)
    error_2 = mean_squared_error(test_prediction, y_test)
    score_2 = model.score(X_test, y_test)
    # print(error, error_2)
    # print(score, score_2)


    model_2 = sm.OLS(y_train, X_train)
    results = model_2.fit()
    predicts = results.predict(X_train)
    print(predicts)
    print(y_train)
    print(results.summary())
