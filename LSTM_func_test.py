import warnings
import chainer
import chainer.functions as F
from chainer import function
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer.training import extensions
from chainer.dataset import concat_examples
import numpy as cp
import copy
from chainer import reporter as reporter_module
import sys
import time

class LSTM_CrossEntropy(L.Classifier):
    def __init__(self, predictor):
        super(LSTM_CrossEntropy, self).__init__(predictor, lossfun=F.mean_squared_error)

    def __call__(self, x, t):
        batch_size = len(x)
        seq_len = len(x[0])

        self.y = None
        self.loss = 0
        # self.accuracy = None
        # acc = []

        for loop in range(batch_size):
            self.predictor.reset_state()
            for i in range(seq_len):
                pred = self.predictor(x[loop][i])
            obs = t[loop]
            self.loss += self.lossfun(pred, obs)
            # acc.append(self.accfun(pred, obs))

        # 各lossの平均を取る
        self.loss /= batch_size
        # self.accuracy = sum(acc) / len(acc)
        # reporter に loss,accuracy の値を渡す
        reporter.report({'loss': self.loss}, self)
        # reporter.report({'accuracy': self.accuracy}, self)

        return self.loss

class LSTM_Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size=100, seq_len=20, support_len=10, repeat=True):
        self.seq_length = seq_len
        self.support_len = support_len
        self.nsamples = len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat

        start = time.time()
        self.x = [dataset[i:i + self.seq_length] for i in range(self.nsamples - self.seq_length)]
        self.t = [[dataset[i]] for i in range(self.seq_length, self.nsamples)]
        print("Cupy allocation time:", time.time() - start)

        self.epoch = 0
        self.iteration = 0
        self.loop = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration >= 1:
            raise StopIteration

        self.iteration += 1
        if self.repeat:
            self.offsets = cp.random.randint(0, self.nsamples - self.seq_length - 1, size=self.batch_size)
            self.offsets = self.offsets.tolist()
        else:
            self.offsets = cp.arange(0, self.nsamples - self.seq_length - 1)
            self.offsets = self.offsets.tolist()

        x, t = self.get_data()
        self.epoch = int((self.iteration * self.batch_size) // self.nsamples)

        return x, t

    def get_data(self):
        x = [self.x[os] for os in self.offsets]
        t = [self.t[os] for os in self.offsets]

        return x, t

    def serialze(self, serialzier):
        self.iteration = serialzier('iteration', self.iteration)
        self.epoch = serialzier('epoch', self.epoch)

    @property
    def epoch_detail(self):
        return self.epoch

    def reset(self):
        self.epoch = 0
        self.iteration = 0
        self.loop = 0