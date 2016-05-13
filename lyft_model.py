import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
rforecast = rpackages.importr('forecast')
import sys

class LyftModel(object):
    """
    Builds several Lyft prediction models
    """

    def __init__(self, filename):
        """
        Input: Organized pandas Dataframe with Uber data
        """
        df = pd.read_csv(filename, parse_dates=['record_time'])
        df.set_index('record_time', inplace=True)
        df.index = df.index - pd.Timedelta(hours=7)
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        df['dayofweek'] = df.index.dayofweek
        df['weekofyear'] = df.index.weekofyear
        df['primetime_percentage'] = df['primetime_percentage'].fillna('0%').apply(lambda x: int(x[:-1]))

        hourly = df.groupby(['date','hour','city','ride_type']).mean().reset_index()
        hourly['record_time'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ":00:00")
        hourly.set_index('record_time', inplace=True)

        features = ['avg_est_price','city','ride_type','eta_seconds','estimated_distance_miles','estimated_duration_seconds','base_charge','cost_minimum','cost_per_mile','cost_per_minute', 'cancel_penalty_amount','num_drivers', 'seats','trust_and_service','primetime_percentage','hour','dayofweek','weekofyear'] #,'lag_1'
        hourly = pd.get_dummies(hourly[features], columns=['city','ride_type','hour','dayofweek']).drop(['city_chicago','hour_0','dayofweek_0'], axis=1) # ,'ride_type_lyft_plus'
        self.df = hourly.dropna()
        self.kfold_indices = []

    def make_holdout_split(self, leaveout=1, weekly=False):
        """
        Output: X_train, X_hold, y_train, y_hold

        Train test split by specified leaveout value
        """
        if weekly:
            # modified to exclude the current week prices
            self.total_folds = self.df['weekofyear'].unique()[:-1]
            print "total weeks:", self.total_folds
            # leaves out the latest week based on the order of the array which might be a problem when it predicts the next year
            lo = self.total_folds[-leaveout:][0]
            train_set = self.df.query("weekofyear < @lo")
            hold_set = self.df.query("weekofyear >= @lo and weekofyear <= @self.total_folds[-1]")
            print "holdout week:",hold_set['weekofyear'].unique()
            self.train_set = train_set.copy()
            self.hold_set = hold_set.copy()
            # print self.train_set.weekofyear.unique()
            # print self.hold_set.weekofyear.unique()
            y_train = train_set.pop("avg_est_price")
            y_hold = hold_set.pop("avg_est_price")

        return train_set.reset_index(),hold_set.reset_index(),y_train.reset_index(),y_hold.reset_index()

    def get_kfold_timeseries_indices(self, X, y, lag=1, ahead=1):
        """
        Output: Train and test indices

        Gets custom train and test indices for each kfold
        """
        kfolds = X['weekofyear'].unique()
        self.num_folds = kfolds.shape[0]-1
        for i in xrange(self.num_folds):
            if i-lag < 0:
                train_set = X.query("weekofyear <= @kfolds[@i]")
                self.train_indices = train_set.index.values
                print 'train_set', train_set.weekofyear.unique()
            else:
                train_set = X.query("weekofyear <= @kfolds[@i] and weekofyear > @kfolds[@i-@lag]")
                self.train_indices = train_set.index.values
                print 'train_set', train_set.weekofyear.unique()
            if i+ahead > self.num_folds:
                test_set = X.query("weekofyear >= @kfolds[@i+1]")
                self.test_indices = test_set.index.values
                print 'test_set', test_set.weekofyear.unique()
            else:
                test_set = X.query("weekofyear >= @kfolds[@i+1] and weekofyear <= @kfolds[@i+@ahead]")
                self.test_indices = test_set.index.values
                print 'test_set', test_set.weekofyear.unique()
            self.kfold_indices.append((self.train_indices, self.test_indices))
        return self.kfold_indices

    def format_guess(self):
        """
        Output: Numpy Array

        Format guess correct for prediction
        """
        city = raw_input("What city would you like to know the prices for? ")
        hour = raw_input("What hour in the future? (e.g. 8,12): ")
        cartype = raw_input("What cartype would you like to take? (e.g. uberX, uberBLACK): ")
        guess = [city, hour, cartype]
        X_g = self.df2[(self.df2[guess[0]] == 1) & (self.df2['hour'] == int(guess[1])) & (self.df2[guess[2]] == 1)]
        X_g.pop('avg_price_est')
        X_g = X_g.mean().values.reshape(1,-1)
        self.X_g = X_g

    def perform_grid_search(self, X_train, X_test, y_train, y_test, estimator, custom_cv, params):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """
        if estimator.__class__.__name__ != "XGBRegressor":
            gridsearch = GridSearchCV(estimator,
                                         params,
                                         n_jobs=-1,
                                         verbose=True,
                                         scoring='mean_squared_error',
                                         cv=custom_cv)
        else:
            gridsearch = GridSearchCV(estimator,
                                         params,
                                         n_jobs=-1,
                                         verbose=True,
                                         cv=custom_cv)

        gridsearch.fit(X_train, y_train)

        print "best parameters {}: {}".format(estimator.__class__.__name__, gridsearch.best_params_)

        best_model = gridsearch.best_estimator_

        y_pred = best_model.predict(X_test)

        print "MSE with best {}: {}".format(estimator.__class__.__name__, mean_squared_error(y_true=y_test, y_pred=y_pred))

        base_est = estimator

        idx = custom_cv[-1][1]
        base_est.fit(X_train.iloc[idx], y_train[idx])
        base_y_pred = base_est.predict(X_test)

        print "MSE with default param:", mean_squared_error(y_true=y_test, y_pred=base_y_pred)

        return gridsearch, best_model, y_pred

    def forecast_with_arima(self, X_hold, y_hold):
        """
        Forecast next week's prices with ARIMA
        """
        train_set = pd.concat([X_hold,y_hold], axis=1)
        self.X_forecast = self.hold_set[self.hold_set.columns[1:]].copy()
        # assumes weekofyear is increasing
        self.X_forecast['weekofyear'] = self.X_forecast['weekofyear'].apply(lambda x: x+1)
        self.X_forecast.index = self.X_forecast.index + pd.Timedelta(days=7)
        test_set = self.X_forecast.reset_index()
        train_name = "data/lyft_train_forecast.csv"
        train_set.to_csv(train_name)
        test_name = "data/lyft_test_forecast.csv"
        test_set.to_csv(test_name)
        rfeatures = ['eta_seconds','primetime_percentage','city_denver','cost_per_mile', 'city_ny', 'city_seattle','city_sf','ride_type_lyft','ride_type_lyft_plus']
        feats = 'c({})'.format(str(rfeatures)[1:-1])
        r = robjects.r("""
        train_set = read.csv("{}")
        test_set = read.csv("{}")
        y = train_set['avg_est_price']
        features = {}
        X = train_set[features]
        X_test = test_set[features]
        fit = auto.arima(y, xreg=X)
        y_pred = forecast(fit, xreg=X_test)
        """.format(train_name, test_name, feats))
        print robjects.r("""y_pred['model']""")
        r_lower = robjects.r("""y_pred['lower']""")[0]
        r_upper = robjects.r("""y_pred['upper']""")[0]
        r_pred = robjects.r("""y_pred['mean']""")[0]
        y_pred = [r_pred[i] for i in range(len(r_pred))]
        self.y_forecast = y_pred
        self.y_forecast = pd.DataFrame(self.y_forecast, index=self.X_forecast.index, columns=['y_forecast'])
        self.y_forecast = pd.concat([self.X_forecast, self.y_forecast], axis=1)
        name = "lyft_arima"
        saved_filename = "rideshare_app/data/{}_forecast.csv".format(name)
        self.y_forecast.to_csv(saved_filename)
        print "saved prediction values to {}".format(saved_filename)

    def run_arima_cv(self, X_train, y_train, custom_cv):
        """
        Cross validate each fold with auto.arima
        """
        train_set = pd.concat([X_train,y_train], axis=1)
        rfeatures = ['eta_seconds','primetime_percentage','city_denver','cost_per_mile', 'city_ny', 'city_seattle','city_sf','ride_type_lyft','ride_type_lyft_plus']
        for i, (train_index, test_index) in enumerate(custom_cv):
            train_fold = train_set.iloc[train_index]
            train_name = "data/lyft_train{}.csv".format(i)
            train_fold.to_csv(train_name)
            test_fold = train_set.iloc[test_index]
            test_name = "data/lyft_test{}.csv".format(i)
            test_fold.to_csv(test_name)
            y_pred = self.run_auto_arima(train_name, test_name, features=rfeatures)
            y_true = test_fold['avg_est_price'].values
            print "ARIMA KFold{}, MSE: {}".format(i, mean_squared_error(y_true, y_pred))

    def run_auto_arima(self, train_name, test_name, features):
        """
        Output: Ndarray

        Returns predictions from auto.arima model
        """
        feats = 'c({})'.format(str(features)[1:-1])
        r = robjects.r("""
        train_set = read.csv("{}")
        test_set = read.csv("{}")
        y = train_set['avg_est_price']
        features = {}
        X = train_set[features]
        X_test = test_set[features]
        fit = auto.arima(y, xreg=X)
        y_pred = forecast(fit, xreg=X_test)
        """.format(train_name, test_name, feats))
        print robjects.r("""y_pred['model']""")
        r_lower = robjects.r("""y_pred['lower']""")[0]
        r_upper = robjects.r("""y_pred['upper']""")[0]
        r_pred = robjects.r("""y_pred['mean']""")[0]
        y_pred = [r_pred[i] for i in range(len(r_pred))]

        return y_pred

        """
        ARIMA(2,1,3)
        AIC=9426.43   AICc=9426.72   BIC=9493.54
        ARIMA KFold0, MSE: 90.9646588096

        ARIMA(5,1,4)
        AIC=17465.2   AICc=17465.44   BIC=17564.2
        ARIMA KFold1, MSE: 240.919514377

        ARIMA(3,0,3) with non-zero mean
        AIC=18589.68   AICc=18589.87   BIC=18677.16
        ARIMA KFold2, MSE: 78.1285872852

        ARIMA(3,1,3)
        """

    def run_linear_models(self, estimator, X_train, y_train, X_hold, y_hold):
        """
        Output: Best Model

        Returns MSE scores for each of the linear models
        """
        estimator.fit(X_train, y_train)
        est_name = estimator.__class__.__name__
        if est_name != 'ElasticNetCV':
            print "best param for {}: {}".format(est_name, estimator.alpha_)
        else:
            print "best param for {}: {}, {}".format(est_name, estimator.alpha_, estimator.l1_ratio_)
        y_pred = estimator.predict(X_hold)
        print "{} MSE: {}".format(est_name, mean_squared_error(y_hold, y_pred))
        self.pickle_model(estimator, name=est_name.lower() + "_lyft")

        if est_name != 'ElasticNetCV':
            self.make_forecast(estimator, name=est_name.lower() + "_lyft", alpha=estimator.alpha_)
        else:
            self.make_forecast(estimator, name=est_name.lower() + "_lyft", alpha=estimator.alpha_, l1_ratio=estimator.l1_ratio_)

    def print_feature_importance(self, X_train, best_rf_model):
        """
        Prints the important features
        """
        for feature in sorted(zip(X_train.columns,best_rf_model.feature_importances_), key=lambda x:x[1])[::-1]:
            print feature

    def pickle_model(self, model, name):
        """
        Output: Saved Model

        Pickles our model for later use
        """
        with open("rideshare_app/data/{}.pkl".format(name), 'w') as f:
            pickle.dump(model, f)
        print "{} is pickled.".format(name)

    def make_forecast(self, model, name, alpha=None, l1_ratio=None):
        """
        Output: DataFrame

        Train on the holdout set and make predictions for the next week
        """
        X_hold = self.hold_set[self.hold_set.columns[1:]]
        y_hold = self.hold_set['avg_est_price']
        if name.split("_")[0] == "ridgecv":
            model = Ridge(alpha=alpha)
        elif name.split("_")[0] == "lassocv":
            model = Lasso(alpha=alpha)
        elif name.split("_")[0] == "elasticnetcv":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_hold, y_hold)
        self.X_forecast = X_hold.copy()
        # assumes weekofyear is increasing
        self.X_forecast['weekofyear'] = self.X_forecast['weekofyear'].apply(lambda x: x+1)
        self.X_forecast.index = self.X_forecast.index + pd.Timedelta(days=7)
        self.y_forecast = model.predict(self.X_forecast)
        self.y_forecast = pd.DataFrame(self.y_forecast, index=self.X_forecast.index, columns=['y_forecast'])
        self.y_forecast = pd.concat([self.X_forecast, self.y_forecast], axis=1)
        saved_filename = "rideshare_app/data/{}_forecast.csv".format(name)
        self.y_forecast.to_csv(saved_filename)
        print "saved prediction values to {}".format(saved_filename)

def plot_prediction_true_res(y_pred):
    """
    Output: Plot of Model Prediction vs. True
    """

    # y_trues = pd.DataFrame(lym.hold_set['avg_price_est'])
    y_preds = pd.DataFrame(y_pred, index=lym.hold_set.index, columns=['y_pred'])
    data = pd.concat([lym.hold_set,y_preds], axis=1)
    # print data.columns
    cities = ['city_denver','city_sf','city_seattle','city_ny','city_chicago']
    cartypes = ['ride_type_lyft','ride_type_lyft_line','ride_type_lyft_plus']
    for city in cities:
        for cartype in cartypes:
            plt.cla()
            if city != 'city_chicago':
                sub_data = data[(data[city] == 1) & (data[cartype] == 1)] #.resample('10T')
            else:
                sub_data = data[(data[cities[0]] == 0) & (data[cities[1]] == 0) & (data[cities[2]] == 0) & (data[cities[3]] == 0) & (data[cartype] == 1)] #.resample('10T')
            fig, ax = plt.subplots(2,1,figsize=(20,10))

            ax[0].plot_date(sub_data.index.to_pydatetime(), sub_data['avg_est_price'].values, 'o--', label='true data');
            ax[0].plot_date(sub_data.index.to_pydatetime(), sub_data['y_pred'].values, '-', label='prediction', alpha=0.8)
            ax[0].xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=2))
            ax[0].xaxis.set_minor_formatter(DateFormatter('%H'))
            ax[0].xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
            ax[0].xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
            ax[0].xaxis.grid(True, which="minor")
            ax[0].set_xlabel('hour')
            ax[0].set_ylabel('average price estimate')
            ax[0].legend(loc="upper right")
            ax[0].set_title("Y Predictions vs Y Trues For {}, {}".format(cartype.split('_')[-1],city.split('_')[-1]))

            data['resid'] = data['avg_est_price'] - data['y_pred']
            if city != 'city_chicago':
                resid = data[(data[city] == 1) & (data[cartype] == 1)]['resid'] #.resample('10T')
            else:
                resid = data[(data[cities[0]] == 0) & (data[cities[1]] == 0) & (data[cities[2]] == 0) & (data[cities[3]] == 0) & (data[cartype] == 1)]['resid'] #.resample('10T')

            ax[1].plot_date(resid.index.to_pydatetime(), resid.values, 'o', label='residuals', alpha=0.3);
            ax[1].xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=2))
            ax[1].xaxis.set_minor_formatter(DateFormatter('%H'))
            ax[1].xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
            ax[1].xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
            ax[1].xaxis.grid(True, which="minor")
            ax[1].set_xlabel('hour')
            ax[1].set_ylabel('price residuals')
            ax[1].legend(loc="upper right")

            plt.tight_layout()
            plt.savefig('plots/pred_int_{}_{}.png'.format(cartype.split('_')[-1],city.split('_')[-1]))
            print "finished plot {}, {}".format(cartype.split('_')[-1],city.split('_')[-1])
            plt.close('all')

if __name__ == '__main__':
    filename = sys.argv[1]  # 'data/organized_uber_41816.csv'
    lym = LyftModel(filename)

    # Subsetting by week -> array([ 7,  8,  9, 10, 11, 12, 13], dtype=int32)

    X_train, X_hold, y_train, y_hold = lym.make_holdout_split(leaveout=1, weekly=True)

    custom_cv = lym.get_kfold_timeseries_indices(X_train, y_train, lag=1, ahead=1)

    X_train.pop('record_time')
    y_train.pop('record_time')
    X_hold.pop('record_time')
    y_hold.pop('record_time')

    ### GridSearchCV for best parameters for RF
    # rf_params = {'n_estimators': [10, 100, 200],
    #                         'criterion': ['mse'],
    #                         'min_samples_split': [2, 4, 6, 7],
    #                         'min_samples_leaf': [1, 2],
    #                         'max_features': ['sqrt',None,'log2']}
    # gridsearch, best_model, y_pred = lym.perform_grid_search(X_train, X_hold, y_train.values.reshape(-1), y_hold.values.reshape(-1), RandomForestRegressor(), custom_cv, rf_params)
    #
    # plot_prediction_true_res(y_pred)
    # lym.pickle_model(best_model, name='lyft_rf_model')
    # lym.make_forecast(best_model, name='lyft_rf_model')
    #
    # xgb_params = {'max_depth': [2,4,6],
    #                         'n_estimators': [50,100,200],
    #                         'gamma': [0,1,2]}
    # gridsearch, best_model, y_pred = lym.perform_grid_search(X_train, X_hold, y_train.values.reshape(-1), y_hold.values.reshape(-1), XGBRegressor(), custom_cv, xgb_params)
    #
    # plot_prediction_true_res(y_pred)
    # lym.pickle_model(best_model, name='lyft_xgboost_model')
    # lym.make_forecast(best_model, name='lyft_xgboost_model')
    #
    # ## Multiple Regression with CV
    # for regression in [RidgeCV(scoring='mean_squared_error', cv=custom_cv), LassoCV(cv=custom_cv, n_jobs=-1), ElasticNetCV(cv=custom_cv, n_jobs=-1)]:
    #     lym.run_linear_models(regression, X_train, y_train.values.reshape(-1), X_hold, y_hold.values.reshape(-1))

    ## ARIMA in R with CV
    lym.run_arima_cv(X_train, y_train, custom_cv)
    lym.forecast_with_arima(X_hold, y_hold)

    ### Cross val score with baseline RF and XGB
    # rf = RandomForestRegressor(n_estimators=10)
    # xgb = XGBRegressor(n_estimators=100)
    # mses = -cross_val_score(estimator=xgb, X=X_train, y=y_train.values.reshape(-1), cv=custom_cv, scoring='mean_squared_error', n_jobs=-1)
    # print "CV on baseline XGB with MSE:", zip(X_train['weekofyear'].unique(),mses)


    # be able to ask your model to predict what the price will be based on the city and hour time of travel and type of transport

    # print 'RF holdout set MSE:', lym.score_model_on_holdout(X_hold, y_hold)
    # lym.format_df_for_guessing(df2)
    # lym.format_guess()
    # print lym.estimator.predict(lym.X_g)

    # lym.print_feature_importance(X_train, best_model)

    """
    MSE with best RandomForestRegressor: 34.3179549988
    MSE with default param: 36.0067202375

    MSE with best XGBRegressor: 32.0875568031
    MSE with default param: 35.6150438829

    RidgeCV MSE: 39.0565059553
    LassoCV MSE: 48.5996541771
    ElasticNetCV MSE: 48.558654884

    ARIMA KFold0, MSE: 30.4710572333
    ARIMA KFold1, MSE: 103.384083234
    ARIMA KFold2, MSE: 23.1263255722

    ARIMA(2,1,2)
    AIC=15808.69   AICc=15808.86   BIC=15890.33
    """
