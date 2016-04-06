def make_train_test_split_kfold(self, X, y, number_folds, estimator, lag=10, ahead=1):
    """
    Output: KFold Training and test split (Time Series)

    Finds the number of kfolds and splits the data appropriately. Lag indicates the amount of folds to keep in training set. "Ahead" indicates the amount of folds to predict for the test set.
    """
    y_preds = []
    y_tests = []
    k = int(np.floor(X.shape[0]/float(number_folds)))
    for i in xrange(1, number_folds + 1):
        if i-lag < 0 or lag == 0:
            X_train = X[:(k*i)]
            y_train = y[:(k*i)]
        else:
            X_train = X[(k*(i-lag)):(k*i)]
            y_train = y[(k*(i-lag)):(k*i)]
        X_test = X[(k*i):(k*(i+ahead))]
        y_test = y[(k*i):(k*(i+ahead))]
