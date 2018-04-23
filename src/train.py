import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
 
import matplotlib.pyplot as plt

import sys

from sklearn.preprocessing import MinMaxScaler

import network
from util import plot_result

import chainer
from chainer import Chain, Variable, datasets, optimizers
from chainer import report, training
from chainer.training import extensions
import chainer.cuda

flg_MT = 1

start = dt.date(2016,1,1)
end = dt.date(2017,9,20)
df= web.DataReader('AMZN',"iex",start,end)
 
print(df.head(5))


data = np.array(df[["open", "close"]])

scaler = MinMaxScaler(feature_range=(0, 1))
#df['close'] = scaler.fit_transform(df['close'])
data = scaler.fit_transform(data)

print(data.shape)

x = []
x_long = []
x_short = []
t = []

if flg_MT == 0:
    n_skip = 1
elif flg_MT == 1:
    n_skip = 3

N = len(df)
M = 25

for n in range(M*n_skip,N):
    if flg_MT == 0:
        _x = data[n-M:n, :]
        _t = data[n, 1]
        x.append(_x)
        t.append(_t)
    elif flg_MT == 1:
        _x_temp = data[n-M*n_skip:n, :] # all data
        #_x_short = [np.array(_x_temp[i*n_skip]) for i in range(M)]
        _x_short = np.array(data[n-M:n,:])
        _x_long = [np.array(_x_temp[i*n_skip:(i+1)*n_skip]).mean(axis=0) for i in range(M)]
        _t = data[n, 1]
        x_long.append(_x_long)
        x_short.append(_x_short)
        t.append(_t)


if flg_MT == 0:
    x = np.array(x, dtype = np.float32)
    t = np.array(t, dtype = np.float32).reshape(len(t),1)
elif flg_MT == 1:
    x_short = np.array(x_short, dtype = np.float32)
    x_long = np.array(x_long, dtype = np.float32)
    #x = list(zip(x_short, x_long))
    x = np.concatenate((x_short, x_long), axis=2)
    t = np.array(t, dtype = np.float32).reshape(len(t),1)
 
# 訓練：60%, 検証：40%で分割する
n_train = int(len(x) * 0.6)
dataset = list(zip(x, t))
train, test = chainer.datasets.split_dataset(dataset, n_train)

# 乱数のシードを固定 (再現性の確保)
np.random.seed(1)
 
# モデルの宣言
if flg_MT == 0:
    model = network.RNN(30, 1)
elif flg_MT == 1:
    #model = network.RNN_MT(30, 1)
    model = network.RNN_MT_ATTENTION(30, 1)
 
# GPU対応
chainer.cuda.get_device(0).use()
model.to_gpu()                 
 
# Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# Iterator
batchsize = 20
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
 
# Updater &lt;- LSTM用にカスタマイズ
updater = network.LSTMUpdater(train_iter, optimizer,device = 0)
 
# Trainerとそのextensions
epoch = 3000
trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
 
# 評価データで評価
trainer.extend(extensions.Evaluator(test_iter, model,device = 0))
 
# 学習結果の途中を表示する
trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
 
# １エポックごとに、trainデータに対するlossと、testデータに対するlossを出力させる
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))

trainer.run()

plot_result.plot()