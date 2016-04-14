import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

class UberModel(object):
    """
    Builds several Uber prediction models
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

        hourly = df.groupby(['date','hour','city','display_name']).mean().reset_index()
        hourly['record_time'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ":00:00")
        hourly.set_index('record_time', inplace=True)

        # df['dayofmonth'] = df.index.day
        # df['trip_minutes'] = df['trip_duration'] / 60.
        # df['avg_price_per_min'] = df['avg_price_est'] / df['trip_minutes']

        # consider changing the cross validation paradigm to increase more lag
        # features to use: trip distance, trip duration, pickup_estimate (maybe)
        # not constant features: 'surge_minimum_price','capacity','base_price', 'base_minimum_price','cost_per_minute', 'cost_per_distance','cancellation_fee', 'service_fee'

        # df['lag_1'] = df['avg_price_est'].diff(periods=1)

        features = ['avg_price_est','city','display_name','trip_duration', 'trip_distance','pickup_estimate','hour','dayofweek','weekofyear'] #,'lag_1','surge_multiplier'
        hourly = pd.get_dummies(hourly[features], columns=['city','display_name','hour','dayofweek']).drop(['city_chicago','display_name_uberASSIST','hour_0','dayofweek_0'], axis=1)
        self.df = hourly.dropna()
        self.kfold_indices = []

    def make_holdout_split(self, leaveout=1, weekly=False):
        """
        Output: X_train, X_hold, y_train, y_hold

        Train test split by specified leaveout value
        """
        if weekly:
            self.total_folds = self.df['weekofyear'].unique()
            # leaves out the latest week based on the order of the array which might be a problem when it predicts the next year
            lo = self.total_folds[-leaveout:][0]
            train_set = self.df.query("weekofyear < @lo")
            hold_set = self.df.query("weekofyear >= @lo")
            self.train_set = train_set.copy()
            self.hold_set = hold_set.copy()
            # print self.train_set.weekofyear.unique()
            # print self.hold_set.weekofyear.unique()
            y_train = train_set.pop("avg_price_est")
            y_hold = hold_set.pop("avg_price_est")

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

    def make_forecast(self, model, name):
        """
        Output: DataFrame

        Train on the holdout set and make predictions for the next week
        """
        X_hold = self.hold_set[self.hold_set.columns[1:]]
        y_hold = self.hold_set['avg_price_est']
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

    def perform_grid_search(self, X_train, X_test, y_train, y_test, estimator, custom_cv, params):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """
        gridsearch = GridSearchCV(estimator,
                                     params,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='mean_squared_error',
                                     cv=custom_cv)

        gridsearch.fit(X_train, y_train)

        print "best parameters {}: {}".format(estimator.__class__.__name__, gridsearch.best_params_)

        best_model = gridsearch.best_estimator_

        y_pred = best_model.predict(X_test)

        print "MSE with best {}: {}".format(estimator.__class__.__name__, mean_squared_error(y_true=y_test, y_pred=y_pred))

        base_est = estimator

        idx = custom_cv[-1][1]
        rf.fit(X_train.iloc[idx], y_train[idx])
        base_y_pred = base_est.predict(X_test)

        print "MSE with default param:", mean_squared_error(y_true=y_test, y_pred=base_y_pred)

        return gridsearch, best_model, y_pred

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

def plot_prediction_true_res(y_pred):
    """
    Output: Plot of Model Prediction vs. True
    """

    # y_trues = pd.DataFrame(ubm.hold_set['avg_price_est'])
    y_preds = pd.DataFrame(y_pred, index=ubm.hold_set.index, columns=['y_pred'])
    data = pd.concat([ubm.hold_set,y_preds], axis=1)
    # print data.columns
    cities = ['city_denver','city_sf','city_seattle','city_ny','city_chicago']
    cartypes = ['display_name_uberX','display_name_uberXL','display_name_uberBLACK','display_name_uberSUV']
    for city in cities:
        for cartype in cartypes:
            plt.cla()
            if city != 'city_chicago':
                sub_data = data[(data[city] == 1) & (data[cartype] == 1)] #.resample('10T')
            else:
                sub_data = data[(data[cities[0]] == 0) & (data[cities[1]] == 0) & (data[cities[2]] == 0) & (data[cities[3]] == 0) & (data[cartype] == 1)] #.resample('10T')
            fig, ax = plt.subplots(2,1,figsize=(20,10))

            ax[0].plot_date(sub_data.index.to_pydatetime(), sub_data['avg_price_est'].values, 'o--', label='true data');
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

            data['resid'] = data['avg_price_est'] - data['y_pred']
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
    filename = 'data/organized_uber_41116.csv'
    ubm = UberModel(filename)

    # Subsetting by week -> array([ 7,  8,  9, 10, 11, 12, 13], dtype=int32)

    X_train, X_hold, y_train, y_hold = ubm.make_holdout_split(leaveout=1, weekly=True)

    custom_cv = ubm.get_kfold_timeseries_indices(X_train, y_train, lag=1, ahead=1)

    X_train.pop('record_time')
    y_train.pop('record_time')
    X_hold.pop('record_time')
    y_hold.pop('record_time')

    ### GridSearchCV for best parameters for RF
    params = {'n_estimators': [10, 100, 200],
                            'criterion': ['mse'],
                            'min_samples_split': [2, 4, 6, 7],
                            'min_samples_leaf': [1, 2],
                            'max_features': ['sqrt',None,'log2']}
    gridsearch, best_model, y_pred = ubm.perform_grid_search(X_train, X_hold, y_train.values.reshape(-1), y_hold.values.reshape(-1), RandomForestRegressor(), custom_cv, params)

    # plot_prediction_true_res(y_pred)

    # ubm.pickle_model(best_model, name='model2_wo_surgemulti')
    #
    # ubm.make_forecast(best_model, name='model2_wo_surgemulti')

    ### Cross val score with baseline RF
    # rf = RandomForestRegressor(n_estimators=10)
    # mses = -cross_val_score(estimator=rf, X=X_train, y=y_train.values.reshape(-1), cv=custom_cv, scoring='mean_squared_error', n_jobs=-1)
    # print "CV on baseline RF with MSE:", zip(X_train['weekofyear'].unique(),mses)


    # be able to ask your model to predict what the price will be based on the city and hour time of travel and type of transport

    # print 'RF holdout set MSE:', ubm.score_model_on_holdout(X_hold, y_hold)
    # ubm.format_df_for_guessing(df2)
    # ubm.format_guess()
    # print ubm.estimator.predict(ubm.X_g)

    # ubm.print_feature_importance(X_train, best_rf_model)

    """
    best parameters: {'max_features': None, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 10}
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
       max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
       min_samples_split=6, min_weight_fraction_leaf=0.0,
       n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
       verbose=0, warm_start=False)
    MSE with best rf: 0.0696791534095
    MSE with default param rf: 0.224983742868

    Without leakage variables:

    best parameters: {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 10.9785314346
    MSE with default param rf: 15.322677252

    Without lag_1 variable:

    best parameters: {'max_features': 'log2', 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'mse', 'n_estimators': 10}
    MSE with best rf: 46.900728035
    MSE with default param rf: 44.0100363882

    With cv lag of 5:

    best parameters: {'max_features': 'log2', 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 46.158427494
    MSE with default param rf: 56.2259376978

    -- try to resample by hour but still include all the categorical variables

    With hourly resampling and cv lag of 2:

    best parameters: {'max_features': 'sqrt', 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 39.3497185302
    MSE with default param rf: 94.9461884732

    With hourly resampling and cv lag of 1:

    best parameters: {'max_features': 'sqrt', 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 200}
    MSE with best rf: 36.5273036401
    MSE with default param rf: 60.2200781633

    -- go back and see what other features can you include that is mostly constant through the week

    Modified holdout set to include week 12 and 13:

    best parameters: {'max_features': None, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 280.266058956
    MSE with default param rf: 304.215061729

    With surge_multiplier:

    best parameters: {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 15.51976143
    MSE with default param rf: 32.4664843472

    ('display_name_uberSUV', 0.29000892463099914)
    ('trip_distance', 0.17669217696686926)
    ('display_name_uberBLACK', 0.1549772144537914)
    ('trip_duration', 0.12605338196387572)
    ('surge_multiplier', 0.071224937402909513)
    ('display_name_uberWAV', 0.064931857129935586)
    ('display_name_uberX', 0.026107336328640759)
    ('display_name_uberXL', 0.020264040684914435)
    ('display_name_uberSELECT', 0.017277217298207182)
    ('city_ny', 0.017257763698162489)
    ('display_name_uberESPANOL', 0.014434691681362639)
    ('city_seattle', 0.0094182122349883279)
    ('pickup_estimate', 0.0040876569743090261)

    Retrieved data up to week 14:

    best parameters: {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 0.942149836266
    MSE with default param rf: 10.9731377293

    Without surge_multiplier:

    best parameters: {'max_features': 'sqrt', 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 100}
    MSE with best rf: 56.7717152006
    MSE with default param rf: 64.234876547

    ('display_name_uberSUV', 0.2048637442374508)
    ('trip_duration', 0.13164410221511086)
    ('pickup_estimate', 0.10503238907363363)
    ('display_name_uberWAV', 0.10330720494351724)
    ('trip_distance', 0.097831781783953228)
    ('display_name_uberX', 0.085968653767959577)
    ('display_name_uberBLACK', 0.055809745694839055)
    ('city_seattle', 0.041266396367005236)
    ('city_ny', 0.035409041602259689)
    ('city_denver', 0.01865789832987827)
    ('display_name_uberXL', 0.018290636329722696)
    ('display_name_uberESPANOL', 0.01731926767473092)
    ('display_name_uberSELECT', 0.016296225095864675)
    ('city_sf', 0.013362568284479584)
    ('display_name_uberFAMILY', 0.011900193110712545)
    ('display_name_uberTAXI', 0.0053023498845518368)
    ('display_name_uberPEDAL', 0.0041246772754761048)
    """

    """
    With surge multiplier, week 14 predictions:

    ('display_name_uberSUV', 0.28156529820578796)
    ('trip_distance', 0.17386787481727198)
    ('display_name_uberBLACK', 0.15105326480490774)
    ('trip_duration', 0.12405091650062343)
    ('surge_multiplier', 0.095744259344206009)
    ('display_name_uberWAV', 0.06128319437327865)
    ('display_name_uberX', 0.026588448626477423)
    ('display_name_uberXL', 0.019741683016970161)
    ('display_name_uberSELECT', 0.016919647962355146)
    ('city_ny', 0.016547926784837016)
    ('display_name_uberESPANOL', 0.015035628528900597)
    ('city_seattle', 0.0092183694111987111)
    ('city_sf', 0.0037929761758299907)
    ('pickup_estimate', 0.0016991033673227963)
    ('display_name_uberFAMILY', 0.0013389520832636254)
    ('city_denver', 0.00029488467750748992)
    ('display_name_uberTAXI', 0.00019888029444719031)
    ('display_name_uberWARMUP', 0.00015433717504211345)
    ('weekofyear', 0.0001082871211898807)
    ('dayofweek_2', 0.00010365204867123911)
    ('dayofweek_4', 9.6276784885273946e-05)
    ('hour_6', 7.7131736607251585e-05)
    ('hour_7', 6.0289433406320183e-05)
    ('display_name_uberPEDAL', 5.6811952373247677e-05)
    ('hour_4', 4.8798261917506261e-05)
    ('dayofweek_3', 4.8087045900860311e-05)
    ('display_name_uberTAHOE', 3.2908466222634051e-05)
    ('hour_17', 2.6315613136821515e-05)
    ('hour_11', 2.6295081924580555e-05)
    ('hour_15', 2.3549520986304168e-05)
    ('hour_5', 2.2416132868638021e-05)
    ('dayofweek_1', 2.1325986219720821e-05)
    ('hour_16', 1.6482223851737059e-05)
    ('hour_12', 1.3402264159827176e-05)
    ('hour_14', 1.3103875331400295e-05)
    ('hour_18', 1.2745491510340055e-05)
    ('hour_2', 1.1210860588588259e-05)
    ('hour_19', 1.1101405245830553e-05)
    ('hour_8', 9.8018692204899525e-06)
    ('dayofweek_6', 9.6259623977638091e-06)
    ('dayofweek_5', 8.8389207543897053e-06)
    ('hour_20', 8.6190750627970933e-06)
    ('hour_13', 7.514480751485061e-06)
    ('hour_3', 7.2361288056986364e-06)
    ('hour_22', 5.630772279482073e-06)
    ('hour_9', 5.3823984167670465e-06)
    ('hour_21', 4.1433315946478496e-06)
    ('hour_1', 3.9336987278618334e-06)
    ('hour_10', 2.3299191862041496e-06)
    ('hour_23', 1.1059855764074888e-06)
    """
