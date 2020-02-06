import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from cleaning_script import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from weather_data import *
from sklearn import preprocessing
from statsmodels.tools import add_constant
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import linear_model
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/route_stops_more.csv')
df_weather = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/houston_weather.csv')

def make_linear_regression(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)

    error = np.sqrt(mean_squared_error(train_prediction, y_train))
    score = model.score(X_train, y_train)

    test_prediction = model.predict(X_test)

    error_2 = np.sqrt(mean_squared_error(test_prediction, y_test))
    score_2 = model.score(X_test, y_test)

    print((error, error_2) , (score, score_2))

    return (error, error_2) , (score, score_2)

def run_ols_model(y_train, X_train):

    X_train = add_constant(X_train)
    model_2 = sm.OLS(y_train, X_train)
    results = model_2.fit()
    print(results.summary())

def random_Forest_regression(X_train, X_test, y_train, y_test):

    regr = RandomForestRegressor(max_depth = 6, n_estimators = 90)
    regr.fit(X_train, y_train)
    predict = regr.predict(X_train)
    predict_2 = regr.predict(X_test)

    return predict, predict_2

def random_forest_classifier(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(max_depth = 7, n_estimators = 100)
    clf.fit(X_train, y_train)
    a = clf.predict(X_train)
    print(clf.score(X_test, y_test))
    print(clf.score(X_train, y_train))
    print(a)

if __name__ == "__main__":
    df, y_data, y_data_2 = make_dataframe(df, df_weather)

    keep = ['number_of_orders','weight_of_orders', 'quantity_of_pieces_on_orders', 'month', 'days',
            'Morning', 'Afternoon', 'rain_fall', 'unloading_speed']

    df = df[keep]
    df = one_hot_encode_columns(df, ['rain_fall','unloading_speed'])
    names = df.columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    print(scaled_data)
    new_df = pd.DataFrame(scaled_data, columns = names)
    print(new_df)
    X_train, X_test, y_train, y_test = train_test_split(new_df, y_data_2.to_numpy(), test_size = 0.20)

    run_ols_model(y_train, X_train)
    make_linear_regression(X_train, X_test, y_train, y_test)

    # model = GradientBoostingRegressor(n_estimators=100, max_depth = 6, min_samples_split=5)
    # model.fit(X_train, y_train)
    # print(model.score(X_train, y_train))
    # print(model.score(X_test, y_test))
    #
    # predict, predict_2 = random_Forest_regression(X_train, X_test, y_train, y_test)
    # print(np.sqrt(mean_squared_error(predict, y_train)))
    # print(np.sqrt(mean_squared_error(predict_2, y_test)))
    #
    # print(predict)
    # print(y_train)

    # clf = linear_model.Lasso(alpha=0.1)
    # clf.fit(X_train, y_train )
    # print(clf.score(X_train, y_train))
    # print(clf.score(X_test, y_test))
    #
    # ridge = linear_model.Ridge(alpha=100000.0)
    # ridge.fit(X_train, y_train)
    # print(ridge.score(X_train, y_train))
    # print(clf.score(X_test, y_test))
