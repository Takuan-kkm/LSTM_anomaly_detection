import warnings
import chainer
import chainer.functions as F
from chainer import function
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer.training import extensions
from chainer.dataset import concat_examples
import cupy as cp
import copy
from chainer import reporter as reporter_module
import sys
import time


class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=None)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        x_batch, t = data_iter.__next__()

        optimizer.target.predictor.reset_state()
        optimizer.target.cleargrads()

        loss = optimizer.target(x_batch, t)  # np:0.25secくらい cp:0.5secくらい

        loss.backward()  # np:0.40secくらい cp:0.87secくらい
        loss.unchain_backward()
        optimizer.update()


class LSTM_MSE(L.Classifier):
    def __init__(self, predictor):
        super(LSTM_MSE, self).__init__(predictor, lossfun=F.mean_squared_error)

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
                pred = self.predictor(x[loop][i].reshape([1, len(x[loop][i])]))
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
        self.x = cp.array([dataset[i:i + self.seq_length] for i in range(self.nsamples - self.seq_length)],
                          dtype="float32")
        self.t = cp.array([[dataset[i]] for i in range(self.seq_length, self.nsamples)], dtype="float32")
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
        else:
            self.offsets = cp.arange(0, self.nsamples - self.seq_length - 1)

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


class MyEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for x_batch, t_batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                with function.no_backprop_mode():
                    eval_func(x_batch, t_batch)

            summary.add(observation)

        return summary.compute_mean()
