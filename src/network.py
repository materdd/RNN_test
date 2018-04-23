# chainerと必要なパッケージをインポート
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, datasets, optimizers
from chainer import report, training
from chainer.training import extensions
#import chainer.cuda
import chainer
import sys
import cupy as cp
 
class RNN(Chain):
    def __init__(self, n_units, n_output):
        super().__init__()
        with self.init_scope():
            self.l1 = L.LSTM(None, n_units)
            self.l2 = L.Linear(None, n_output)
        
    def reset_state(self):
        self.l1.reset_state()
 
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss},self)
        return loss
        
    def predict(self, x):
        if chainer.config.train:
            h1 = F.dropout(self.l1(x),ratio = 0.5)
        else:
            h1 = self.l1(x)
        return self.l2(h1)

class RNN_MT(Chain):
    def __init__(self, n_units, n_output):
        super().__init__()
        with self.init_scope():
            self.l1_s = L.LSTM(None, n_units)
            self.l1_l = L.LSTM(None, n_units)
            self.l2 = L.Linear(None, n_output)

        #self.train = train
        
    def reset_state(self):
        self.l1_s.reset_state()
        self.l1_l.reset_state()
 
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss},self)
        return loss
        
    def predict(self, x):
        #x_s = x[:,:,0:2]
        #x_l = x[:,:,2:4]
        x_s = x
        x_l = x

        if chainer.config.train:
            h1_s = F.dropout(self.l1_s(x_s),ratio = 0.5)
            h1_l = F.dropout(self.l1_l(x_l),ratio = 0.5)
            h1 = F.concat((h1_s, h1_l), axis=1)
        else:
            h1_s = self.l1_s(x_s)
            h1_l = self.l1_l(x_l)
            h1 = F.concat((h1_s, h1_l), axis=1)
        return self.l2(h1)


class RNN_MT_ATTENTION(Chain):
    def __init__(self, n_units, n_output):
        super().__init__()
        with self.init_scope():
            self.atten_fc_x = L.Linear(None, n_units)
            self.atten_fc_c = L.Linear(None, n_units)
            self.atten_fc = L.Linear(None, 4)
            self.l1 = L.LSTM(None, n_units)
            self.l2 = L.Linear(None, n_output)

        self.n_units = n_units
        
    def reset_state(self):
        self.l1.reset_state()
 
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss},self)
        return loss
        
    def predict(self, x):
        if chainer.config.train:
            if self.l1.c is None:
                xp = cp
                batch = len(x)
                self.l1.c = Variable(xp.zeros((batch, self.n_units), dtype=x[0].dtype))
            
            atten_x = F.dropout(self.atten_fc_x(x), ratio=0.5)
            atten_c = F.dropout(self.atten_fc_c(self.l1.c), ratio=0.5)
            temp_atten = F.concat((atten_x, atten_c), axis=1)
            
            atten = F.softmax(self.atten_fc(temp_atten))

            print(x.data)
            x_atten = F.math.basic_math.mul(x, atten)
            print(x_atten.data)
            h1 = F.dropout(self.l1(x_atten),ratio = 0.5)
            #h1 = F.concat((h1_s, h1_l), axis=1)
        else:
            h1 = self.l1(x)
            #h1 = F.concat((h1_s, h1_l), axis=1)
        return self.l2(h1)

## LSTMUpdater
class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater,self).__init__(data_iter, optimizer, device=None)
        self.device = device
        
    def update_core(self):
        data_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        
        batch = data_iter.__next__()
        x_batch, t_batch = chainer.dataset.concat_examples(batch, self.device)
        
        optimizer.target.reset_state()           #追加
        optimizer.target.cleargrads()
        loss = optimizer.target(x_batch, t_batch)
        loss.backward()
        loss.unchain_backward()                  #追記
        optimizer.update() 