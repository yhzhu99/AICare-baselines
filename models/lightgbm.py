import lightgbm as lgb


class LightGBM():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        param = {'num_leaves': 31, 'objective': 'binary'}
        param['metric'] = 'auc'
        self.param = param

        self.model = None

    def fit(self, x, y):
        train_data = lgb.Dataset(x, label=y[:, 0])
        num_round = 100
        self.model = lgb.train(self.param, train_data, num_round)
    def predict(self, x):
        return self.model.predict(x)