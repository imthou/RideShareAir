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
from collections import OrderedDict

class RideShareModel(object):
    """
    Builds several Ride Share prediction models
    """

    def __init__(self, filename):
        """
        Input: Organized pandas Dataframe with RideShare data
        """
        self.filename = filename
        self.df = pd.read_csv(self.filename, parse_dates=['record_time'])
        self.df.set_index('record_time', inplace=True)
        self.df.index = self.df.index - pd.Timedelta(hours=7)
        self.df['hour'] = self.df.index.hour
        self.df['date'] = self.df.index.date
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['weekofyear'] = self.df.index.weekofyear
        if 'lyft' in self.filename:
            self.df['primetime_percentage'] = self.df['primetime_percentage'].fillna('0%').apply(lambda x: int(x[:-1]))
            self.features = ['avg_est_price','city','ride_type','eta_seconds','estimated_distance_miles','estimated_duration_seconds','base_charge','cost_minimum','cost_per_mile','cost_per_minute', 'cancel_penalty_amount','num_drivers', 'seats','trust_and_service','primetime_percentage','hour','dayofweek','weekofyear']
            self.cartypes = ['ride_type_lyft','ride_type_lyft_line','ride_type_lyft_plus']
            self.hourly = self.df.groupby(['date','hour','city','ride_type']).mean().reset_index()
            self.company = "_lyft"
        else:
            self.features = ['avg_price_est','city','display_name','trip_duration', 'trip_distance','pickup_estimate','surge_multiplier','hour','dayofweek','weekofyear'] #,'lag_1'
            self.cartypes = ['display_name_uberX','display_name_uberXL','display_name_uberBLACK','display_name_uberSUV']
            self.hourly = self.df.groupby(['date','hour','city','display_name']).mean().reset_index()
            self.company = "_uber"

        self.hourly['record_time'] = pd.to_datetime(self.hourly['date'].astype(str) + ' ' + self.hourly['hour'].astype(str) + ":00:00")
        self.hourly.set_index('record_time', inplace=True)
        self.cities = ['city_denver','city_sf','city_seattle','city_ny','city_chicago']

        if 'lyft' in self.filename:
            self.hourly = pd.get_dummies(self.hourly[self.features], columns=['city','ride_type','hour','dayofweek']).drop(['city_chicago','hour_0','dayofweek_0'], axis=1) # ,'ride_type_lyft_plus'
        else:
            self.hourly = pd.get_dummies(self.hourly[self.features], columns=['city','display_name','hour','dayofweek']).drop(['city_chicago','display_name_uberASSIST','hour_0','dayofweek_0'], axis=1)
        self.df = self.hourly.dropna()
        self.kfold_indices = []
        self.model_results = OrderedDict()
        self.model_params = OrderedDict()

    def _make_holdout_split(self, leaveout=1, weekly=False):
        """
        Input: Leaveout Fold, Weekly Boolean
        Output: X_train, X_hold, y_train, y_hold

        Train test split by specified leaveout value
        """
        if weekly:
            # modified to exclude the current week prices
            self.total_folds = self.df['weekofyear'].unique()
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
            if 'lyft' in self.filename:
                y_train = train_set.pop("avg_est_price")
                y_hold = hold_set.pop("avg_est_price")
            else:
                y_train = train_set.pop("avg_price_est")
                y_hold = hold_set.pop("avg_price_est")

        self.X_train, self.X_hold, self.y_train, self.y_hold = train_set.reset_index(),hold_set.reset_index(),y_train.reset_index(),y_hold.reset_index()

    def _get_kfold_timeseries_indices(self, lag=1, ahead=1):
        """
        Input: Lag Interval, Ahead Interval
        Output: Train and test indices

        Gets custom train and test indices for each kfold
        """
        self.kfolds = self.X_train['weekofyear'].unique()
        self.num_folds = self.kfolds.shape[0]-1
        for i in xrange(self.num_folds):
            if i-lag < 0: # prevents errors when lag is greater than the available data avaliable at that fold
                train_set = self.X_train.query("weekofyear <= @self.kfolds[@i]")
                self.train_indices = train_set.index.values
                print 'train_set', train_set.weekofyear.unique()
            else:
                train_set = self.X_train.query("weekofyear <= @self.kfolds[@i] and weekofyear > @self.kfolds[@i-@lag]")
                self.train_indices = train_set.index.values
                print 'train_set', train_set.weekofyear.unique()
            if i+ahead > self.num_folds: # prevents errors when ahead is greater than the n folds
                test_set = self.X_train.query("weekofyear >= @self.kfolds[@i+1]")
                self.test_indices = test_set.index.values
                print 'test_set', test_set.weekofyear.unique()
            else:
                test_set = self.X_train.query("weekofyear >= @self.kfolds[@i+1] and weekofyear <= @self.kfolds[@i+@ahead]")
                self.test_indices = test_set.index.values
                print 'test_set', test_set.weekofyear.unique()
            self.kfold_indices.append((self.train_indices, self.test_indices))
        print "number of kfolds:", len(self.kfold_indices)

    def _pop_record_time(self):
        """
        Output: training and test data without without record time
        """
        self.X_train_nort = self.X_train.copy()
        self.X_train_nort.pop('record_time')
        self.y_train_nort = self.y_train.copy()
        self.y_train_nort.pop('record_time')
        self.X_hold_nort = self.X_hold.copy()
        self.X_hold_nort.pop('record_time')
        self.y_hold_nort = self.y_hold.copy()
        self.y_hold_nort.pop('record_time')

    def _run_models(self):
        """
        Run all models and export model
        """
        self._make_holdout_split(leaveout=1, weekly=True)

        self._get_kfold_timeseries_indices(lag=1, ahead=1)

        self._pop_record_time()

        # ### GridSearchCV for best parameters for RF
        self.rf_params = {'n_estimators': [10, 100, 200],
                                'criterion': ['mse'],
                                'min_samples_split': [2, 4, 6, 7],
                                'min_samples_leaf': [1, 2],
                                'max_features': ['sqrt',None,'log2']}
        self._perform_grid_search(RandomForestRegressor(), self.rf_params)

        self._plot_prediction_true_res()
        if 'lyft' in self.filename:
            self._pickle_model(self.best_model, name='lyft_rf_model')
            self._make_forecast(self.best_model, name='lyft_rf_model')
        else:
            self._pickle_model(self.best_model, name='model1_w_surgemulti')
            self._make_forecast(self.best_model, name='model1_w_surgemulti')

        self.xgb_params = {'max_depth': [2,4,6],
                                'n_estimators': [50,100,200],
                                'gamma': [0,1,2]}
        self._perform_grid_search(XGBRegressor(), self.xgb_params)

        # self._plot_prediction_true_res()
        if 'lyft' in self.filename:
            self._pickle_model(self.best_model, name='lyft_xgboost_model')
            self._make_forecast(self.best_model, name='lyft_xgboost_model')
        else:
            self._pickle_model(self.best_model, name='xgboost_model')
            self._make_forecast(self.best_model, name='xgboost_model')

        # Multiple Regression with CV
        for regression in [RidgeCV(scoring='mean_squared_error', cv=self.kfold_indices, alphas=np.logspace(-3,1,100)), LassoCV(cv=self.kfold_indices, n_jobs=-1, alphas=np.logspace(-3,1,100)), ElasticNetCV(cv=self.kfold_indices, n_jobs=-1, l1_ratio=np.linspace(0.001,1,100), alphas=np.logspace(-3,1,100))]:
            self._run_linear_models(regression)

        # ARIMA in R with CV
        self._run_arima_cv()
        self._forecast_with_arima()


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

    def _perform_grid_search(self, estimator, params):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """
        if estimator.__class__.__name__ != "XGBRegressor":
            self.gridsearch = GridSearchCV(estimator,
                                         params,
                                         n_jobs=-1,
                                         verbose=True,
                                         scoring='mean_squared_error',
                                         cv=self.kfold_indices)
            self.gridsearch.fit(self.X_train_nort, self.y_train_nort.values.reshape(-1))
        else:
            self.gridsearch = GridSearchCV(estimator,
                                         params,
                                         n_jobs=-1,
                                         verbose=True,
                                         cv=self.kfold_indices)

            self.gridsearch.fit(self.X_train_nort, self.y_train_nort.values.reshape(-1))

        self.model_params[estimator.__class__.__name__] = self.gridsearch.best_params_
        print "best parameters {}: {}".format(estimator.__class__.__name__, self.gridsearch.best_params_)

        self.best_model = self.gridsearch.best_estimator_

        self.y_pred = self.best_model.predict(self.X_hold_nort)

        results = mean_squared_error(y_true=self.y_hold_nort.values.reshape(-1), y_pred=self.y_pred)
        self.model_results[estimator.__class__.__name__] = results
        print "MSE with best {}: {}".format(estimator.__class__.__name__, results)

        self.base_est = estimator

        idx = self.kfold_indices[-1][1]
        self.base_est.fit(self.X_train_nort.iloc[idx], self.y_train_nort.values.reshape(-1)[idx])
        self.base_y_pred = self.base_est.predict(self.X_hold_nort)

        print "MSE with default param:", mean_squared_error(y_true=self.y_hold_nort.values.reshape(-1), y_pred=self.base_y_pred)

    def _forecast_with_arima(self):
        """
        Forecast next week's prices with ARIMA
        """
        train_set = pd.concat([self.X_hold_nort,self.y_hold_nort], axis=1)
        self.X_forecast = self.hold_set[self.hold_set.columns[1:]].copy()
        # assumes weekofyear is increasing
        self.X_forecast['weekofyear'] = self.X_forecast['weekofyear'].apply(lambda x: x+1)
        self.X_forecast.index = self.X_forecast.index + pd.Timedelta(days=7)
        test_set = self.X_forecast.reset_index()
        if 'lyft' in self.filename:
            train_name = "data/lyft_train_forecast.csv"
            test_name = "data/lyft_test_forecast.csv"
        else:
            train_name = "data/uber_train_forecast.csv"
            test_name = "data/uber_test_forecast.csv"
        train_set.to_csv(train_name)
        test_set.to_csv(test_name)
        print "forecast train_set size:", train_set.shape
        print "forecast test_set size:", test_set.shape
        try:
            if 'lyft' in self.filename:
                r = robjects.r("""
                train_set = read.csv("{}")
                test_set = read.csv("{}")
                y = train_set['avg_est_price']
                features = {}
                X = train_set[features]
                X_test = test_set[features]
                fit = auto.arima(y, xreg=X)
                y_pred = forecast(fit, xreg=X_test)
                """.format(train_name, test_name, self.feats))
            else:
                r = robjects.r("""
                train_set = read.csv("{}")
                test_set = read.csv("{}")
                y = train_set['avg_price_est']
                features = {}
                X = train_set[features]
                X_test = test_set[features]
                fit = auto.arima(y, xreg=X)
                y_pred = forecast(fit, xreg=X_test)
                """.format(train_name, test_name, self.feats))
            print robjects.r("""y_pred['model']""")
            self.model_params["ARIMA"] = robjects.r("""y_pred['method']""")[0][0]
            self.r_lower = robjects.r("""y_pred['lower']""")[0]
            self.r_upper = robjects.r("""y_pred['upper']""")[0]
            self.r_pred = robjects.r("""y_pred['mean']""")[0]
            self.y_pred = [self.r_pred[i] for i in range(len(self.r_pred))]
            self.y_forecast = self.y_pred
            self.y_forecast = pd.DataFrame(self.y_forecast, index=self.X_forecast.index, columns=['y_forecast'])
            self.y_forecast = pd.concat([self.X_forecast, self.y_forecast], axis=1)
            if 'lyft' in self.filename:
                self.arima_name = "lyft_arima"
            else:
                self.arima_name = "uber_arima"
            self.saved_arima_filename = "rideshare_app/data/{}_forecast.csv".format(self.arima_name)
            self.y_forecast.to_csv(self.saved_arima_filename)
            print "saved prediction values to {}".format(self.saved_arima_filename)
        except:
            print "No suitable ARIMA model found for forecasting"

    def _run_arima_cv(self):
        """
        Cross validate each fold with auto.arima
        """
        train_set = pd.concat([self.X_train_nort,self.y_train_nort], axis=1)
        if 'lyft' in self.filename:
            self.rfeatures = ['eta_seconds','primetime_percentage','city_denver','cost_per_mile', 'city_ny', 'city_seattle','city_sf','ride_type_lyft','ride_type_lyft_plus']
        else:
            self.rfeatures = ['trip_duration', 'trip_distance', 'pickup_estimate', 'surge_multiplier','city_denver', 'city_ny', 'city_seattle','city_sf','display_name_uberBLACK','display_name_uberSUV','display_name_uberX','display_name_uberXL'] # display_name_uberTAXI, display_name_uberSELECT
        self.feats = 'c({})'.format(str(self.rfeatures)[1:-1])
        print "number of kfolds:", len(self.kfold_indices)
        for i, (train_index, test_index) in enumerate(self.kfold_indices):
            # if i <= 12:
            #     continue
            train_fold = train_set.iloc[train_index]
            print "size of training index {}:  {}".format(i, len(train_index))
            # if len(train_index) < 4000:
            #     print "insufficient training sample size"
            #     continue
            if 'lyft' in self.filename:
                train_name = "data/lyft_train{}.csv".format(i)
                test_name = "data/lyft_test{}.csv".format(i)
            else:
                train_name = "data/uber_train{}.csv".format(i)
                test_name = "data/uber_test{}.csv".format(i)
            train_fold.to_csv(train_name)
            test_fold = train_set.iloc[test_index]
            print "size of testing index {}:  {}".format(i, len(test_index))
            # if len(test_index) < 4000:
            #     print "insufficient testing sample size"
            #     continue
            test_fold.to_csv(test_name)
            try:
                self._run_auto_arima(train_name, test_name)
                if 'lyft' in self.filename:
                    self.y_true = test_fold['avg_est_price'].values
                else:
                    self.y_true = test_fold['avg_price_est'].values
                results = mean_squared_error(self.y_true, self.y_pred)
                self.model_results['ARIMA_{}'.format(i)] = results
                print "ARIMA KFold{}, MSE: {}".format(i, results)
            except:
                print "No Suitable ARIMA model found"
                continue

    def _run_auto_arima(self, train_name, test_name):
        """
        Output: Ndarray

        Returns predictions from auto.arima model
        """
        if 'lyft' in self.filename:
            self.r = robjects.r("""
            train_set = read.csv("{}")
            test_set = read.csv("{}")
            y = train_set['avg_est_price']
            features = {}
            X = train_set[features]
            X_test = test_set[features]
            fit = auto.arima(y, xreg=X)
            y_pred = forecast(fit, xreg=X_test)
        """.format(train_name, test_name, self.feats))
        else:
            self.r = robjects.r("""
            train_set = read.csv("{}")
            test_set = read.csv("{}")
            y = train_set['avg_price_est']
            features = {}
            X = train_set[features]
            X_test = test_set[features]
            fit = auto.arima(y, xreg=X)
            y_pred = forecast(fit, xreg=X_test)
        """.format(train_name, test_name, self.feats))
        print robjects.r("""y_pred['model']""")
        self.r_lower = robjects.r("""y_pred['lower']""")[0]
        self.r_upper = robjects.r("""y_pred['upper']""")[0]
        self.r_pred = robjects.r("""y_pred['mean']""")[0]
        self.y_pred = [self.r_pred[i] for i in range(len(self.r_pred))]

    def _run_linear_models(self, estimator):
        """
        Output: Best Model

        Returns MSE scores for each of the linear models
        """
        estimator.fit(self.X_train_nort, self.y_train_nort.values.reshape(-1))
        self.est_name = estimator.__class__.__name__
        if self.est_name != 'ElasticNetCV':
            print "best param for {}: {}".format(self.est_name, estimator.alpha_)
            self.model_params[self.est_name] = estimator.alpha_
        else:
            print "best param for {}: {}, {}".format(self.est_name, estimator.alpha_, estimator.l1_ratio_)
            self.model_params[self.est_name] = [estimator.alpha_, estimator.l1_ratio_]
        self.y_pred = estimator.predict(self.X_hold_nort)
        results = mean_squared_error(self.y_hold_nort.values.reshape(-1), self.y_pred)
        self.model_results[self.est_name] = results
        print "{} MSE: {}".format(self.est_name, results)
        self._pickle_model(estimator, name=self.est_name.lower() + self.company)

        if self.est_name != 'ElasticNetCV':
            self._make_forecast(estimator, name=self.est_name.lower() + self.company, alpha=estimator.alpha_)
        else:
            self._make_forecast(estimator, name=self.est_name.lower() + self.company, alpha=estimator.alpha_, l1_ratio=estimator.l1_ratio_)

    def print_feature_importance(self):
        """
        Prints the important features
        """
        for feature in sorted(zip(self.X_train_nort.columns,self.best_model.feature_importances_), key=lambda x:x[1])[::-1]:
            print feature

    def _pickle_model(self, model, name):
        """
        Output: Saved Model

        Pickles our model for later use
        """
        with open("rideshare_app/pkl_models/{}.pkl".format(name), 'w') as f:
            pickle.dump(model, f)
        print "{} is pickled.".format(name)

    def _make_forecast(self, model, name, alpha=None, l1_ratio=None):
        """
        Output: DataFrame

        Train on the holdout set and make predictions for the next week
        """
        X_hold = self.hold_set[self.hold_set.columns[1:]]
        if 'lyft' in self.filename:
            y_hold = self.hold_set['avg_est_price']
        else:
            y_hold = self.hold_set['avg_price_est']
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

    def _plot_prediction_true_res(self):
        """
        Output: Plot of Model Prediction vs. True
        """

        # y_trues = pd.DataFrame(ubm.hold_set['avg_price_est'])
        self.y_preds = pd.DataFrame(self.y_pred, index=self.hold_set.index, columns=['y_pred'])
        self.data = pd.concat([self.hold_set,self.y_preds], axis=1)
        # print data.columns

        for city in self.cities:
            for cartype in self.cartypes:
                plt.cla()
                if city != 'city_chicago':
                    self.sub_data = self.data[(self.data[city] == 1) & (self.data[cartype] == 1)]
                else:
                    self.sub_data = self.data[(self.data[self.cities[0]] == 0) & (self.data[self.cities[1]] == 0) & (self.data[self.cities[2]] == 0) & (self.data[self.cities[3]] == 0) & (self.data[cartype] == 1)]
                fig, ax = plt.subplots(2,1,figsize=(20,10))
                if 'lyft' in self.filename:
                    ax[0].plot_date(self.sub_data.index.to_pydatetime(), self.sub_data['avg_est_price'].values, 'o--', label='true data')
                else:
                    ax[0].plot_date(self.sub_data.index.to_pydatetime(), self.sub_data['avg_price_est'].values, 'o--', label='true data')
                ax[0].plot_date(self.sub_data.index.to_pydatetime(), self.sub_data['y_pred'].values, '-', label='prediction', alpha=0.8)
                ax[0].xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=2))
                ax[0].xaxis.set_minor_formatter(DateFormatter('%H'))
                ax[0].xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
                ax[0].xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
                ax[0].xaxis.grid(True, which="minor")
                ax[0].set_xlabel('hour')
                ax[0].set_ylabel('average price estimate')
                ax[0].legend(loc="upper right")
                ax[0].set_title("Y Predictions vs Y Trues For {}, {}".format(cartype.split('_')[-1],city.split('_')[-1]))

                if 'lyft' in self.filename:
                    self.data['resid'] = self.data['avg_est_price'] - self.data['y_pred']
                else:
                    self.data['resid'] = self.data['avg_price_est'] - self.data['y_pred']
                if city != 'city_chicago':
                    self.resid = self.data[(self.data[city] == 1) & (self.data[cartype] == 1)]['resid']
                else:
                    self.resid = self.data[(self.data[self.cities[0]] == 0) & (self.data[self.cities[1]] == 0) & (self.data[self.cities[2]] == 0) & (self.data[self.cities[3]] == 0) & (self.data[cartype] == 1)]['resid']

                ax[1].plot_date(self.resid.index.to_pydatetime(), self.resid.values, 'o', label='residuals', alpha=0.3);
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
    # filename = 'data/uber_merged_62016.csv'
    rsm = RideShareModel(filename)
    rsm._run_models()
    for k,v in rsm.model_results.iteritems():
        print "Model: {},       MSE: {}".format(k,v)
    for k,v in rsm.model_params.iteritems():
        print "Model: {},       params: {}".format(k,v)

    """ 6-20-16
    Uber:
        best parameters RandomForestRegressor: {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 10}
        MSE with best RandomForestRegressor: 12.258573567
        MSE with default param: 19.4791162034

        best parameters XGBRegressor: {'n_estimators': 200, 'max_depth': 4, 'gamma': 2}
        MSE with best XGBRegressor: 17.2513617447
        MSE with default param: 30.1662634383

        best param for RidgeCV: 0.001
        RidgeCV MSE: 90.4906253044
        best param for LassoCV: 0.001
        LassoCV MSE: 90.4133364876
        best param for ElasticNetCV: 0.001, 1.0
        ElasticNetCV MSE: 90.4133364876

        ARIMA KFold0, MSE: 209.41649901
        ARIMA KFold1, MSE: 220.637841507
        ARIMA KFold2, MSE: 241.211876874
        ARIMA KFold3, MSE: 298.040536076
        ARIMA KFold4, MSE: 272.382357559
        ARIMA KFold5, MSE: 198.33950856
        ARIMA KFold6, MSE: 273.476545115
        ARIMA KFold7, MSE: 224.937488711
        ARIMA KFold8, MSE: 242.966603036
        ARIMA KFold9, MSE: 186.266007076
        ARIMA KFold10, MSE: 106.252699493
        ARIMA KFold11, MSE: 89.3662169457
        ARIMA KFold12, MSE: 110.89071845

        ARIMA KFold13, MSE: 122.361710872
        ARIMA KFold14, MSE: 218.38183815
        ARIMA KFold15, MSE: 224.359820751

        ARIMA(5,0,5) with non-zero mean
        AIC=41158.28   AICc=41158.5   BIC=41317.13

    Lyft:
        best parameters RandomForestRegressor: {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 200}
        MSE with best RandomForestRegressor: 9.01330786547
        MSE with default param: 12.3387309831

        best parameters XGBRegressor: {'n_estimators': 100, 'max_depth': 6, 'gamma': 1}
        MSE with best XGBRegressor: 9.12105065966
        MSE with default param: 12.258879546

        best param for RidgeCV: 10.0
        RidgeCV MSE: 24.7817085376
        best param for LassoCV: 0.351119173422
        LassoCV MSE: 26.9842571967
        best param for ElasticNetCV: 0.151991108295, 0.001
        ElasticNetCV MSE: 25.4088216239

        ARIMA KFold0, MSE: 30.4710572333
        ARIMA KFold1, MSE: 103.384083234
        ARIMA KFold2, MSE: 23.1263255722
        ARIMA KFold3, MSE: 49.7183093638
        ARIMA KFold4, MSE: 41.5722764037
        ARIMA KFold5, MSE: 139.904041974
        ARIMA KFold6, MSE: 109.803295817
        ARIMA KFold7, MSE: 46.1923982834
        ARIMA KFold8, MSE: 66.2179242989

        ARIMA(2,1,3)
        AIC=15899.27   AICc=15899.46   BIC=15986.74
    """
