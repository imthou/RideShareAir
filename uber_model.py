import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

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
        df['day'] = df.index.day
        df['date'] = df.index.date
        df['dayofweek'] = df.index.dayofweek
        df['weekofyear'] = df.index.weekofyear
        # df['trip_minutes'] = df['trip_duration'] / 60.
        features = ['city','display_name','trip_duration', 'trip_distance','avg_price_est','surge_minimum_price','surge_multiplier','capacity', 'base_price', 'base_minimum_price', 'cost_per_minute', 'cost_per_distance', 'cancellation_fee', 'hour', 'dayofweek','weekofyear']
        df = df[features]
        # df['avg_price_per_min'] = df['avg_price_est'] / df['trip_minutes']
        df = pd.get_dummies(df, columns=['city','display_name'])
        self.df = df.dropna()
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

    def score_model_on_holdout(self, X_hold, y_hold):
        """
        Output: MSE

        Score our model based on holdout set
        """
        self.y_pred = self.estimator.predict(X_hold)
        return mean_squared_error(y_true=y_hold, y_pred=self.y_pred)

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

    def format_df_for_guessing(self, df2):
        """
        Output: DataFrame

        Format the dataframe for prediction
        """
        self.ho = df2.iloc[ubm.split:].reset_index()
        self.ho = self.ho.join(pd.DataFrame(self.y_pred, columns=['y_pred']))
        n_cols = []
        for column in df2.columns:
            if 'display_name_' in column:
                n_cols.append(column.split('display_name_')[1])
            elif 'city_' in column:
                n_cols.append(column.split('city_')[1])
            else:
                n_cols.append(column)
        df2.columns = n_cols
        self.df2 = df2

    def perform_grid_search_rf(self, X_train, X_test, y_train, y_test, custom_cv):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """

        random_forest_grid = {'n_estimators': [10, 100, 200],
                                'criterion': ['mse'],
                                'min_samples_split': [2, 4, 6, 7],
                                'min_samples_leaf': [1, 2],
                                'max_features': ['sqrt',None,'log2']}

        rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                                     random_forest_grid,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='mean_squared_error',
                                     cv=custom_cv)

        rf_gridsearch.fit(X_train, y_train)

        print "best parameters:", rf_gridsearch.best_params_

        best_rf_model = rf_gridsearch.best_estimator_

        y_pred = best_rf_model.predict(X_test)

        print "MSE with best rf:", mean_squared_error(y_true=y_test, y_pred=y_pred)

        rf = RandomForestRegressor()

        idx = custom_cv[-1][1]
        rf.fit(X_train.iloc[idx], y_train[idx])
        base_y_pred = rf.predict(X_test)

        print "MSE with default param rf:", mean_squared_error(y_true=y_test, y_pred=base_y_pred)

        return rf_gridsearch, best_rf_model, y_pred

        """
        best parameters: {'max_features': None, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 10}
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=6, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
        MSE with best rf: 0.0696791534095
        MSE with default param rf: 0.224983742868
        """

def plot_prediction_true_res(y_pred):
    """
    Output: Plot of Model Prediction vs. True
    """

    # y_trues = pd.DataFrame(ubm.hold_set['avg_price_est'])
    y_preds = pd.DataFrame(y_pred, index=ubm.hold_set.index, columns=['y_pred'])
    data = pd.concat([ubm.hold_set,y_preds], axis=1)
    # print data.columns
    cities = ['city_denver','city_sf','city_seattle','city_chicago','city_ny']
    cartypes = ['display_name_uberX','display_name_uberXL','display_name_uberBLACK','display_name_uberSUV']
    for city in cities:
        for cartype in cartypes:
            plt.cla()
            sub_data = data[(data[city] == 1) & (data[cartype] == 1)].resample('10T')
            fig, ax = plt.subplots(2,1,figsize=(20,10))

            ax[0].plot_date(sub_data.index.to_pydatetime(), sub_data['avg_price_est'].values, 'o--', label='true data');
            ax[0].plot_date(sub_data.index.to_pydatetime(), sub_data['y_pred'].values, '-', label='prediction', alpha=0.8)
            ax[0].xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
            ax[0].xaxis.set_minor_formatter(DateFormatter('%H'))
            ax[0].xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
            ax[0].xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
            ax[0].xaxis.grid(True, which="minor")
            ax[0].set_xlabel('hour')
            ax[0].set_ylabel('average price estimate')
            ax[0].legend(loc="upper right")
            ax[0].set_title("Y Predictions vs Y Trues For {}, {}".format(cartype.split('_')[-1],city.split('_')[-1]))

            data['resid'] = data['avg_price_est'] - data['y_pred']
            resid = data[(data[city] == 1) & (data[cartype] == 1)]['resid'].resample('10T')
            ax[1].plot_date(resid.index.to_pydatetime(), resid.values, 'o', label='residuals', alpha=0.3);
            ax[1].xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
            ax[1].xaxis.set_minor_formatter(DateFormatter('%H'))
            ax[1].xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
            ax[1].xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
            ax[1].xaxis.grid(True, which="minor")
            ax[1].set_xlabel('hour')
            ax[1].set_ylabel('price residuals')
            ax[1].legend(loc="upper right")

            plt.tight_layout()
            plt.savefig('plots/pred_int_{}_{}.png'.format(cartype.split('_')[-1],city.split('_')[-1]))
            plt.close('all')

if __name__ == '__main__':
    filename = 'data/organized_uber.csv'
    ubm = UberModel(filename)

    # Subsetting by week -> array([ 7,  8,  9, 10, 11, 12, 13], dtype=int32)

    X_train, X_hold, y_train, y_hold = ubm.make_holdout_split(leaveout=1, weekly=True)

    custom_cv = ubm.get_kfold_timeseries_indices(X_train, y_train, lag=1, ahead=1)
    rf = RandomForestRegressor(n_estimators=10)
    X_train.pop('record_time')
    y_train.pop('record_time')
    X_hold.pop('record_time')
    y_hold.pop('record_time')

    ### GridSearchCV for best parameters for RF
    rf_gridsearch, best_rf_model, y_pred = ubm.perform_grid_search_rf(X_train, X_hold, y_train.values.reshape(-1), y_hold.values.reshape(-1), custom_cv)

    plot_prediction_true_res(y_pred)



    ### Cross val score with baseline RF
    # mses = -cross_val_score(estimator=rf, X=X_train, y=y_train.values.reshape(-1), cv=custom_cv, scoring='mean_squared_error', n_jobs=-1)
    # print "CV on baseline RF with MSE:", zip(X_train['weekofyear'].unique(),mses)


    # ubm.estimate_with_kfold(X_train, y_train, custom_cv, best_rf)
    # plt.close("all")



    # be able to ask your model to predict what the price will be based on the city and hour time of travel and type of transport

    # print 'RF holdout set MSE:', ubm.score_model_on_holdout(X_hold, y_hold)
    # ubm.format_df_for_guessing(df2)
    # ubm.format_guess()
    # print ubm.estimator.predict(ubm.X_g)

    # for feature in sorted(zip(X_train.columns,best_rf_model.feature_importances_), key=lambda x:x[1])[::-1]:
    #     print feature
    """
    ('surge_minimum_price', 0.65043672112895989)
    ('trip_distance', 0.11235756811775592)
    ('base_minimum_price', 0.071450278052160837)
    ('trip_duration', 0.047432957778742471)
    ('surge_multiplier', 0.034332547690270152)
    ('city_chicago', 0.024567895868857846)
    ('base_price', 0.018963870464960432)
    ('display_name_uberFAMILY', 0.017424171748303525)
    ('cost_per_minute', 0.010999549185609681)
    ('cost_per_distance', 0.004134905177000733)
    ('display_name_uberSELECT', 0.0030500966024416083)
    ('cancellation_fee', 0.001522139675136759)
    ('capacity', 0.0010951072457122924)
    ('display_name_uberXL', 0.00057176839751634369)
    ('display_name_uberPEDAL', 0.00045068101017040237)
    ('display_name_uberBLACK', 0.00041177417973595762)
    ('hour', 0.00022524063810545856)
    ('city_sf', 0.00020022229976842952)
    ('city_seattle', 0.00010769207884227673)
    ('weekofyear', 8.5535660336395644e-05)
    ('city_ny', 6.4366108496060823e-05)
    ('dayofweek', 6.3377211253662363e-05)
    ('city_denver', 4.3400773728544319e-05)
    ('display_name_uberWARMUP', 4.3102648635231049e-06)
    ('display_name_uberX', 3.2076590778188309e-06)
    ('display_name_uberSUV', 5.6279930623436635e-07)
    ('display_name_uberWAV', 4.2436189333563819e-08)
    ('display_name_uberESPANOL', 6.1808657326089895e-09)
    ('display_name_uberASSIST', 3.5658315670680356e-09)
    ('display_name_uberTAXI', 0.0)
    ('display_name_uberTAHOE', 0.0)
    """
