import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



####################
# https://www.tensorflow.org/tutorials/keras/regression?hl=ja
####################

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
# check data types
# pd.set_option('display.max_rows', None)
# print(train.dtypes)

###################################
# encode labels to number
###################################
for i in range(train.shape[1]):
    if train.iloc[:, i].dtypes == object:
        le = LabelEncoder()
        le.fit(list(train.iloc[:, i].values) + list(test.iloc[:, i].values))
        train.iloc[:, i] = le.transform(list(train.iloc[:, i].values))
        test.iloc[:, i] = le.transform(list(test.iloc[:, i].values))

###################################
# search for missing data
###################################
# import missingno as msno
# msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
# plt.show()

###################################
# split data and drop unnecessary data
###################################
train_id = train['Id']
test_id = test['Id']

x_train = train.drop(['Id', 'SalePrice'], axis=1)
y_train = train['SalePrice']
x_test = test.drop('Id', axis=1)

###################################
# dealing with missing data
###################################
# x_train = x_train.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
# x_test = x_test.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
# x_train = x_train.fillna(x_train.median())
# x_test = x_test.fillna(x_test.median())
Xmat = pd.concat([x_train, x_test])
Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_test.median())
Xmat = Xmat.fillna(Xmat.median())

x_train['TotalSF'] = x_train['TotalBsmtSF'] + x_train['1stFlrSF'] + x_train['2ndFlrSF']
x_test['TotalSF'] = x_test['TotalBsmtSF'] + x_test['1stFlrSF'] + x_test['2ndFlrSF']

###################################
# 回帰の場合、ターゲットが正規分布に従っていることが重要
# 今回は従っていないので対数を取って擬似的に近づける
###################################
y_train = np.log(y_train)
# ax = sns.distplot(y_train)
# plt.show()

###################################
# 70以上の特徴からランダムフォレストを使って何が重要化推測する
###################################
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(x_train, y_train)

ranking = np.argsort(-rf.feature_importances_)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.barplot(x=rf.feature_importances_[ranking], y=x_train.columns.values[ranking], orient='h')
# ax.set_xlabel("feature importance")
# plt.tight_layout()
# plt.show()

x_train = x_train.iloc[:, ranking[:10]]
x_test = x_test.iloc[:, ranking[:10]]
###################################
# 新しくInteractionを作成
###################################
x_train["Interaction"] = x_train["TotalSF"]*x_train["OverallQual"]
x_test["Interaction"] = x_test["TotalSF"]*x_test["OverallQual"]


###################################
# Interactionを入れてもせいぜい31個しかFeatureがないので、vs SalePriceをすべてプロットしてみましょう。
###################################
# fig = plt.figure(figsize=(12, 7))
# for i in np.arange(10):
#     ax = fig.add_subplot(5, 6, i+1)
#     sns.regplot(x=x_train.iloc[:, i], y=y_train)
#
# plt.tight_layout()
# plt.show()

# 上記で見つかった外れ値を除外する
Xmat = x_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice']<12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea']>5) & (Xmat['SalePrice']<13)].index)
Xmat = Xmat.drop(Xmat[(Xmat['BsmtFinSF1']>5000) & (Xmat['SalePrice']<12.5)].index)

# recover
y_train = Xmat['SalePrice']
x_train = Xmat.drop(['SalePrice'], axis=1)


def xgboost():
    print("Parameter optimization")
    xgb_model = XGBRegressor()
    reg_xgb = GridSearchCV(xgb_model,
                       {'max_depth': [2,4,6],
                        'n_estimators': [50,100,200]}, verbose=1)
    reg_xgb.fit(x_train, y_train)

    return reg_xgb


def nn():
    def create_model(optimizer='adam'):
        model = Sequential()
        model.add(layers.Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(16, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
    optimizer = ['SGD','Adam']
    batch_size = [10, 30, 50]
    epochs = [10, 50, 100]
    param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
    reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    reg_dl.fit(x_train, y_train)

    return reg_dl

def svr():
    reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    reg_svr.fit(x_train, y_train)

    return reg_svr


reg_xgb = xgboost()
reg_dl = nn()
reg_svr = svr()

# これらの予測値を列にまとめ、新しい、第２の訓練マトリックスを作ります。
x_train2 = pd.DataFrame({
    'XGB': reg_xgb.predict(x_train),
    'DL': reg_dl.predict(x_train).ravel(),
    'SVR': reg_svr.predict(x_train)
})

# 上5行だけ見てみると、どのモデルも比較的近い値を予測していますね。
# では、最も単純な線形モデルを使って、各モデルの重みを決定し、最終的な全体の予測値を計算しましょう。最後に、ターゲットをlogにしていたので、expを使って元のスケールに戻すことを忘れないでください。

# second-feature modeling using linear regression
reg = LinearRegression()
reg.fit(x_train2, y_train)

# prediction using the test set
x_test2 = pd.DataFrame({
    'XGB': reg_xgb.predict(x_test),
    'DL': reg_dl.predict(x_test).ravel(),
    'SVR': reg_svr.predict(x_test),
})

# Don't forget to convert the prediction back to non-log scale
y_pred = np.exp(reg.predict(x_test2))

# submission
submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": y_pred
})

submission.to_csv('houseprice.csv', index=False)
