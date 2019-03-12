import lightgbm as lgb
import numpy as np


gbm = lgb.Booster(model_file='./model/lgb.model')
print(gbm.feature_importance())
print('='*100)
print(gbm.feature_importance('gain'))
print('='*100)


def ret_feat_impt(gbm):
    gain = gbm.feature_importance('gain').reshape(-1, 1) / sum(gbm.feature_importance('gain'))
    col = np.array(gbm.feature_name()).reshape(-1, 1)
    return sorted(np.column_stack((col, gain)),key=lambda x: x[1],reverse=True)


importances = ret_feat_impt(gbm)
for x in importances:
    print(x)
