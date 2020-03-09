import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np
# Just for debugging, later delete
import pdb

# ========== Boosting based Decoders ==========


class BaseBoostClass():
    '''Common methods across boosting algorithms

    '''

    def __init__(self, args):
        self.args = args
        self.models = []

    def predict(self, testX):
        predicted = []
        for i in range(self.args.num_chan_kin):
            temp = self.models[i].predict(testX)
            predicted.append(temp)

        predicted = np.array(predicted).T
        return predicted


class XGBoost(BaseBoostClass):
    def __init__(self, args):
        super().__init__(args)

    def train(self, trainX, trainY):
        param = {'objective': "reg:linear",
                 'booster': 'dart',
                 'eval_metric': "rmse",
                 'min_child_weight': self.args.xgb_min_child_weight,
                 'max_depth': self.args.xgb_max_depth,
                 'eta': self.args.xgb_eta,
                 'seed': 0,
                 'silent': True,
                 'gpu_id': 0
                 }
        if self.args.xgb_gpu:
            param['tree_method'] = 'gpu_hist'
        # Creating a model for each joint
        models = []
        for i in range(self.args.num_chan_kin):
            model = xgb.XGBRegressor(**param)
            model.fit(trainX, trainY[:, i])
            models.append(model)
        self.models = models


class LightGBM(BaseBoostClass):
    def __init__(self, args):
        super().__init__(args)

    def train(self, trainX, trainY):
        params = {
            'boosting_type': 'dart',
            'objective': 'regression',
            'metric': 'l2',
            'max_depth': self.args.lgb_max_depth,
            'num_leaves': self.args.lgb_num_leaves,
            'learning_rate': self.args.lgb_learning_rate,
            'min_data_in_leaf': self.args.lgb_min_data_in_leaf,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 0
        }
        if self.args.lgb_gpu:
            params['device_type'] = 'gpu'
        models = []
        for i in range(self.args.num_chan_kin):
            # Create dataset for lightgbm (but for each output)
            lgb_train = lgb.Dataset(trainX, trainY[:, i])
            model = lgb.train(params, lgb_train)
            models.append(model)
        self.models = models


class CatBoost(BaseBoostClass):
    def __init__(self, args):
        super().__init__(args)

    def train(self, trainX, trainY):
        models = []
        for i in range(self.args.num_chan_kin):
            model = CatBoostRegressor(learning_rate=self.args.cb_lr,
                                      loss_function='RMSE',
                                      random_seed=0,
                                      depth=self.args.cb_depth,
                                      l2_leaf_reg=self.args.cb_l2,
                                      has_time=True,
                                      task_type="GPU",
                                      silent=True)
            model.fit(trainX, trainY[:, i])
            models.append(model)
        self.models = models
