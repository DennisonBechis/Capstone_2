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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')
df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')

def make_dataframe(df, df_weather):
    df = main(df)
    df_weather = grouping_by_date_prec(df_weather)
    df_weather = weather_date_to_datetime(df_weather)
    df = pd.merge(df, df_weather, how='left', on="DATE")
    df = df.fillna(0)
    drop_columns = ['arrival_time','departure_time','route_stop_id','left_time',
                    'total_days','arrival_time_1','departure_time_1','grouped_hour',
                    'DATE', 'unplanned', 'stop_type','arrival_driver_id','departure_driver_id',
                    'bill_of_lading_id','stop_number','Night','total_seconds', 'total_minutes','Unnamed:_0']
    y_data = df['grouped_hour']
    df.drop(drop_columns, axis =1, inplace=True)

    return df, y_data

def make_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)

    error = mean_squared_error(train_prediction, y_train)
    score = model.score(X_train, y_train)

    test_prediction = model.predict(X_test)

    error_2 = mean_squared_error(test_prediction, y_test)
    score_2 = model.score(X_test, y_test)

    print((error, error_2) , (score, score_2))

    return (error, error_2) , (score, score_2)

def run_ols_model(y_train, X_train):
    X_train = add_constant(X_train)
    model_2 = sm.OLS(y_train, X_train)
    results = model_2.fit()
    print(results.summary())

def random_Forester_regression(X_train, y_train, X_test, y_test):

    regr = RandomForestRegressor(max_depth = 10, n_estimators = 100)
    regr.fit(X_train, y_train)
    print(regr.score(X_test, y_test))
    print(regr.score(X_train, y_train))

def random_forester_classifier(X_train, y_train, X_test, y_test):

    clf = RandomForestClassifier(max_depth = 10, n_estimators = 100)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.score(X_train, y_train))

if __name__ == "__main__":
    df, y_data = make_dataframe(df, df_weather)

    print(y_data)

    scaler = StandardScaler()
    scaler.fit(df)
    print(scaler.mean_)
    scaled_data = scaler.transform(df)
    print(scaled_data)


    X_train, X_test, y_train, y_test = train_test_split(df, y_data.to_numpy(), test_size = 0.20)

    # make_linear_regression(X_train, X_test, y_train, y_test)
    #
    # run_ols_model(y_train, X_train)
    #
    # random_Forester_regression(X_train, y_train, X_test, y_test)
    #
    # random_forester_classifier(X_train, y_train, X_test, y_test)
