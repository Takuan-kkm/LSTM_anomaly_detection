import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import pickle
from LSTM_func import LSTM_MSE
from LSTM_func import LSTM_Iterator
from LSTM_func import LSTMUpdater
from LSTM_func import MyEvaluator


# Network definition
# LSTM
class LSTM(chainer.Chain):
    def __init__(self, n_in=75, n_units=500, n_units2=400, n_out=75, train=True):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(in_size=n_in, out_size=n_units, lateral_init=chainer.initializers.Normal(scale=0.01))
            self.l2 = L.Linear(in_size=n_units, out_size=n_out, initialW=chainer.initializers.Normal(scale=0.01))
            self.l3 = L.Swish(beta_shape=n_out)
            self.train = train

    def __call__(self, x):
        with chainer.using_config('train', self.train):
            h = self.l1(x)
            h = self.l2(h)
            y = self.l3(h)
        return y

    def reset_state(self):
        self.l1.reset_state()


def main():
    parser = argparse.ArgumentParser(description='Chainer LSTM Network')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='Device specifier. Either ChainerX device '
                             'specifier or an integer. If non-negative integer, '
                             'CuPy arrays with specified device id are used. If '
                             'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                             ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    # parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--plot', type=bool, default=True, help='Disable PlotReport extension')
    parser.add_argument("--train_path", type=str, default="train.sav")
    parser.add_argument("--test_path", type=str, default="test.sav")
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.autotune = True
    # Load dataset
    with open(args.train_path, "rb") as f:
        train = pickle.load(f)
    with open(args.test_path, "rb") as f:
        test = pickle.load(f)

    print("train:", len(train), "test:", len(test))

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    train_iter = LSTM_Iterator(train, args.batchsize, seq_len=20)
    test_iter = LSTM_Iterator(test, args.batchsize, seq_len=20, repeat=False)

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # model = LSTM_CrossEntropy(LSTM(train.shape[1], args.unit, train.shape[1]))
    model = LSTM_MSE(LSTM(75, 200, 400, 75))
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = LSTMUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(MyEvaluator(test_iter, model, device=device))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    # Take a snapshot each ``frequency`` epoch, delete old stale
    # snapshots and automatically load from snapshot files if any
    # files are already resident at result directory.
    trainer.extend(extensions.snapshot(num_retain=1, autoload=args.autoload),
                   trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png')
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume is not None:
        # Resume from a snapshot (Note: this loaded model is to be
        # overwritten by --autoload option, autoloading snapshots, if
        # any snapshots exist in output directory)
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
