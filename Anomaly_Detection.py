import chainer
import chainer.links as L
from chainer import serializers
import LSTM
import argparse
import pickle


class AnomaryDeteciton():
    def __init__(self, model, iterator):
        self.model = model
        self.iterator = iterator

    def predict(self):
        y = []
        t = []
        while ():
            try:
                lookback, obs = self.iterator.__next__()

                self.model.reset_state()
                for i in range(self.iterator.lb_len()):
                    pred = self.model.predictor(lookback[i])
                y.append(pred)
                t.append(obs)
            except StopIteration:
                break

        return y, t

    def calc_error(self, y, t):
        error = []
        if len(y) != len(t):
            return 1
        if len(y[0].shape[0]) != len(t[0].shape[0]):
            return 1

        for i in range(len(y)):
            e = y[i]-t[i]
            error.append(e)

        return error

    def calc_MahalanobisDistanse(self):
        return 0


class Iterator():
    def __init__(self, dataset, lookback_len, seq_len):
        self.dataset = dataset
        self.lookback_len = lookback_len
        self.seq_len = seq_len
        self.n_samples = len(dataset)
        self.pos = 0

    def __next__(self):
        if self.pos + self.lookback_len + self.seq_len > self.n_samples:
            raise StopIteration

        lookback = self.dataset[self.pos:self.pos + self.lookback_len]
        t = self.dataset[self.pos + self.lookback_len:self.pos + self.lookback_len + self.seq_len]
        self.pos += self.seq_len
        return lookback, t

    def lb_len(self):
        return self.lookback_len

    def seq_len(self):
        return self.seq_len


def main():
    parser = argparse.ArgumentParser(description="Chainer Anomary Detection by LSTM")
    parser.add_argument("--model_path", default="result/snapshot_iter_1130")
    parser.add_argument("--keypoints_path", default="test.sav")
    parser.add_argument("--lookback_len", default=20)
    parser.add_argument("--pred_len", default=5)
    args = parser.parse_args()

    model = L.Classifier(LSTM.LSTM())
    # serializers.load_npz(args.model_path, model)
    # with open(args.keypoints_path, "rb") as f:
    #    keypoints = pickle.load(f)

    list = [i for i in range(11)]
    it = Iterator(list, 3, 1)
    print(iter.__next__())
    detector = AnomaryDeteciton(model=model, iterator=it)


if __name__ == '__main__':
    main()
