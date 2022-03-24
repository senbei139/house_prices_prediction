import numpy as np
import pandas as pd
import itertools
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental import preprocessing


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
# pd.set_option('display.max_rows', None)

for i in range(train.shape[1]):
    if train.iloc[:, i].dtypes == object:
        le = LabelEncoder()
        le.fit(list(train.iloc[:, i].values) + list(test.iloc[:, i].values))
        train.iloc[:, i] = le.transform(list(train.iloc[:, i].values))
        test.iloc[:, i] = le.transform(list(test.iloc[:, i].values))

train_id = train['Id']
test_id = test['Id']

train_x = train.drop(['Id', 'SalePrice'], axis=1)
train_y = train['SalePrice']
test_x = test.drop('Id', axis=1)

# dealing with missing data
train_x = train_x.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
test_x = test_x.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
train_x = train_x.fillna(train_x.median())
test_x = test_x.fillna(test_x.median())

train_y = np.log(train_y)
# ax = sns.distplot(train_y)
# plt.show()

# rf = RandomForestRegressor(n_estimators=80, max_features='auto')
# rf.fit(train_x, train_y)

# ranking = np.argsort(-rf.feature_importances_)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.barplot(x=rf.feature_importances_[ranking], y=train_x.columns.values[ranking], orient='h')
# ax.set_xlabel("feature importance")
# plt.tight_layout()
# plt.show()

# train_x = train_x.iloc[:, ranking[:30]]
# test_x = test_x.iloc[:, ranking[:30]]

# fig = plt.figure(figsize=(12, 7))
# for i in np.arange(30):
#     ax = fig.add_subplot(5, 6, i+1)
#     sns.regplot(x=train_x.iloc[:, i], y=train_y)

# plt.tight_layout()
# plt.show()

train_stats = train.describe()
train_stats = train_stats["SalePrice"]
train_stats = train_stats.transpose()


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_x = norm(train_x)
normed_test_x = norm(test_x)


def build_model():

    model_input = layers.Input(shape=(len(train_x.keys())))
    x = model_input
    x = layers.Dense(64, activation='relu', input_shape=[len(train_x.keys())])(x)
    x = layers.Dense(64, activation='relu')(x)
    model_output = layers.Dense(1)(x)

    model = Model(model_input, model_output, name="model_1")
    model.summary()
    keras.utils.plot_model(model, "model_1.png", show_shapes=True)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        metrics=["mae", "mse"],
    )
    return model


model = build_model()

train_dataset = tf.data.Dataset.from_tensor_slices((normed_train_x, train_y))
train_dataset = train_dataset.shuffle(buffer_size=1460).batch(73)
for epoch in range(1000):
    print(f"epoch: {epoch}")

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        result = model.fit(x_batch_train, y_batch_train, verbose=0)
        hist = pd.DataFrame(result.history)
        hist['epoch'] = result.epoch
        hist.tail()

    if epoch % 100 == 0:
        test_scores = model.evaluate(x_batch_train, y_batch_train, verbose=0)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label="Val Error")
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label="Val Error")
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


# plot_history(history)
