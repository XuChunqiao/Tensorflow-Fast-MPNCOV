from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import matplotlib.pyplot as plt
import time
import warnings
import tensorflow as tf
import argparse
import numpy as np
import scipy.io as sio
from utils import AverageMeter, plot_curve, Logger

from trainingFromScratch.imagenet.imagenet_dataset import *
from trainingFromScratch.cifar100.cifar_dataset import *
from src.network import *
from src.representation import *
from model_init import *
from finetune.fine_grained_benchmark.datasets import *




parser = argparse.ArgumentParser(description='TensorFlow MPNCOV Training')
parser.add_argument('dataset', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--benchmark', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--num-classes', default=None, type=int,
                    help='image class number')
parser.add_argument('--train-num', default=None, type=int,
                    help='image class number')
parser.add_argument('--val-num', default=None, type=int,
                    help='image class number')

parser.add_argument('--arch', '-a', metavar='ARCH', default='mpncovresnet50',
                    help='model architecture: ')
parser.add_argument('--representation', default=None, type=str,
                    help='define the representation method:{GAvP, MPNCOV, BCNN, CBP}')
parser.add_argument('--freezed-layer', default=None, type=int,
                    help='freeze layer')


parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='whether use pre-trained model')
parser.add_argument('--model-path', metavar='DIR', default=None,
                    help='path to weights of pretrained model')

parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epoches to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--batchSize', '-b', default=None, type=int, metavar='N',
                    help='mini-batch size(default: 64)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum(default:0.9)')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, metavar='W',
                    help='weight decay(default: 1e-4)')
parser.add_argument('--learning-rate', '-lr', default=None, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--learning-rate-schedule', nargs='+', type=int,  default=[],
                    help='Schedule of learning rate decay')
parser.add_argument('--learning-rate-multiplier', nargs='+', type=float, default=[],
                    help='Schedule of learning rate decay')
parser.add_argument('--WarmingUp', action='store_true',
                    help='whether use warming up')
parser.add_argument('--fc-factor', default=None, type=int, metavar='N',
                    help='define the multiply factor of classifier')



parser.add_argument('--seed', default=None, type=int,
                    help='seed for innitializing training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='number of first stage epoches to run')
parser.add_argument('--exp-dir', metavar='DIR', default='./training_checkpoints',
                    help='path to experiment result')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers(default:4)')

class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, epochs, enable_function, model, batch_size, strategy):
    self.epochs = epochs
    self.batch_size = batch_size
    self.batch_time = AverageMeter()
    self.enable_function = enable_function
    self.strategy = strategy
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    self.train_top1_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.train_top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name='train_top5_accuracy')
    self.val_top1_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    self.val_top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name='test_top5_accuracy')
    self.model = model

  def decay(self, epoch):
    staged_lr = [args.learning_rate * x for x in args.learning_rate_multiplier]

    if args.WarmingUp:
        decay_rate = (args.learning_rate * epoch / args.learning_rate_schedule[0])
    else:
        decay_rate = args.learning_rate

    for st_lr, start_epoch in zip(staged_lr, args.learning_rate_schedule):
        decay_rate = tf.where(epoch < start_epoch, decay_rate, st_lr)
    return decay_rate

  def compute_loss(self, label, predictions):
    predictions = - tf.nn.log_softmax(predictions)
    one_hot_labels = tf.one_hot(label, args.num_classes)
    per_sample_loss = tf.reduce_sum(one_hot_labels * predictions, axis=1)

    # compute cross_entropy loss
    loss = tf.reduce_sum(per_sample_loss) * (1. / self.batch_size)

    # compute regularization loss
    if args.arch.startswith('vgg') and \
            (args.representation == 'BCNN' or args.representation == 'CBP'):
        reg_loss = [tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'classifier' not in v.name]
        if len(reg_loss):
            reg_loss = args.weight_decay * tf.add_n(reg_loss)/self.strategy.num_replicas_in_sync
        else:
            reg_loss = 0
    else:
        reg_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(v)for v in self.model.trainable_variables]) \
                   / self.strategy.num_replicas_in_sync

    loss += reg_loss
    return loss, reg_loss

  def train_step(self, inputs, optimizer):
    """One train step.

    Args:
      inputs: one batch input.

    Returns:
      loss: Scaled loss.
    """

    image, label = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)
      loss, reg_loss = self.compute_loss(label, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)

    # set different learningRate for FC layer
    if args.fc_factor is not None:
        for l in range(len(gradients)-2, len(gradients)):
            gradients[l] = gradients[l] * args.fc_factor

    # update parameters
    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    self.train_top1_metric.update_state(label, predictions)
    self.train_top5_metric.update_state(label, predictions)
    return loss-reg_loss

  def test_step(self, inputs):
    """One test step.

    Args:
      inputs: one batch input.
    """
    image, label = inputs
    if len(image.shape) > 4:
        image = tf.reshape(image, [-1, *image.shape[2:]])
    predictions = self.model(image, training=False)
    if predictions.shape[0] != label.shape[0]:
        s = predictions.shape[0] // label.shape[0]
        predictions = tf.reshape(predictions, [label.shape[0], s, predictions.shape[1]])
        predictions = tf.reduce_mean(predictions, axis=1)

    loss, reg_loss = self.compute_loss(label, predictions)

    self.val_top1_metric.update_state(label, predictions)
    self.val_top5_metric.update_state(label, predictions)
    return loss-reg_loss

  def custom_loop(self, epoch, optimizer, train_dist_dataset, test_dist_dataset,
                  strategy):
    """Custom training and testing loop.

    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(self.train_step,
                                                          args=(dataset_inputs, optimizer,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    def distributed_test_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(self.test_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
    if self.enable_function:
        distributed_train_step = tf.function(distributed_train_step)
        distributed_test_step = tf.function(distributed_test_step)

    self.train_top1_metric.reset_states()
    self.train_top5_metric.reset_states()
    self.val_top1_metric.reset_states()
    self.val_top5_metric.reset_states()
    self.batch_time.reset()

    optimizer.learning_rate = self.decay(epoch)
    print('learningRate: {:.4f}'.format(optimizer.learning_rate.numpy()))
    train_total_loss = 0.0
    num_train_batches = 0.0
    for one_batch in train_dist_dataset:
        end = time.time()
        if args.WarmingUp:
            if epoch < args.learning_rate_schedule[0]:
                batch_learning_rate = self.decay(epoch) + float(num_train_batches / np.ceil(args.train_num/args.batchSize))\
                                      * args.learning_rate / args.learning_rate_schedule[0]
                optimizer.learning_rate = batch_learning_rate
                # print('learningRate: {:.4f}'.format(optimizer.learning_rate.numpy()))

        train_total_loss += distributed_train_step(one_batch)
        num_train_batches += 1
        self.batch_time.update(time.time() - end)
        if num_train_batches % args.print_freq == 0:
            print('learningRate: {:.4f}'.format(optimizer.learning_rate.numpy()))
            template = ('Epoch: {}({}/{})\tTime:{:.4f}({:.4f})\tLoss: {:.4f}\tTop1_Accuracy: {:.4f}\tTop5_Accuracy: {:.4f}')
            print(template.format(epoch+1, int(num_train_batches), int(np.ceil(args.train_num/args.batchSize)),
                                  self.batch_time.val, self.batch_time.avg,
                                  train_total_loss / num_train_batches,
                                  100 * self.train_top1_metric.result(),
                                  100 * self.train_top5_metric.result()))
    self.batch_time.reset()

    val_total_loss = 0.0
    num_val_batches = 0.0
    for one_batch in test_dist_dataset:
        end = time.time()
        val_total_loss += distributed_test_step(one_batch)
        num_val_batches += 1
        self.batch_time.update(time.time() - end)
        if num_val_batches % args.print_freq == 0:
            template = ('Val: {}({}/{})\tTime:{:.4f}({:.4f})\tLoss: {:.4f}\tTop1_Accuracy: {:.4f}\tTop5_Accuracy: {:.4f}')
            print(template.format(epoch+1, int(num_val_batches), int(np.ceil(args.val_num/args.batchSize)),
                                  self.batch_time.val, self.batch_time.avg,
                                  val_total_loss / num_val_batches,
                                  100 * self.val_top1_metric.result(),
                                  100 * self.val_top5_metric.result()))

    return (train_total_loss / num_train_batches,
            100 * self.train_top1_metric.result().numpy(),
            100 * self.train_top5_metric.result().numpy(),
            val_total_loss / num_val_batches,
            100 * self.val_top1_metric.result().numpy(),
            100 * self.val_top5_metric.result().numpy())
def main():
    global args, best_acc, best_epoch
    args = parser.parse_args()
    best_acc = 0.0
    best_epoch = 0
    # make director for store checkpoint files
    if os.path.exists(args.exp_dir) is not True:
        os.mkdir(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, 'logging.txt'))
    print(args)

    gpus = tf.config.experimental.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])

    # Save results for plotting
    results = {'train': {'loss': [], 'top1': [], 'top5': []},
               'val': {'loss': [], 'top1': [], 'top5': []}}

    # optionally resume from a checkpoint
    if args.start_epoch != 0:
        if os.path.isfile(os.path.join(args.exp_dir, 'stats.mat')):
            result = sio.loadmat(os.path.join(args.exp_dir, 'stats.mat'))
            results['train']['loss'] = [] + np.ndarray.tolist(result['train_loss'][0][:args.start_epoch])
            results['train']['top1'] = [] + np.ndarray.tolist(result['train_top1'][0][:args.start_epoch])
            results['train']['top5'] = [] + np.ndarray.tolist(result['train_top5'][0][:args.start_epoch])
            results['val']['loss'] = [] + np.ndarray.tolist(result['val_loss'][0][:args.start_epoch])
            results['val']['top1'] = [] + np.ndarray.tolist(result['val_top1'][0][:args.start_epoch])
            results['val']['top5'] = [] + np.ndarray.tolist(result['val_top5'][0][:args.start_epoch])
            best_acc = max(results['val']['top1'])
            best_epoch = np.where(np.asarray(results['val']['top1']) == best_acc)[0][0]+1
            print('==============================================>BestEpoch:{}\tBest_acc:{:.4f}'.format(best_epoch, best_acc))
            del result
        if os.path.isfile(os.path.join(args.exp_dir, 'checkpoint')):
            file_object = open(os.path.join(args.exp_dir, 'checkpoint'), mode='w')
            all_the_text = 'model_checkpoint_path: "ckpt-{}"\n' \
                           'all_model_checkpoint_paths: "ckpt-{}"\n'.format(str(args.start_epoch), str(args.start_epoch))
            file_object.write(all_the_text)
            file_object.close()

    if 'CUB' in args.benchmark:
        dataset = CUB_dataset
    elif 'Aircrafts' in args.benchmark:
        dataset = Aircrafts_dataset
    elif 'Cars' in args.benchmark:
        dataset = Cars_dataset
    elif 'imagenet' in args.benchmark:
        dataset = ImageNetInput
    elif 'cifar100' in args.benchmark:
        dataset = CifarDataSet
    else:
        raise (RuntimeError('benchmark is not in {CUB, Cars, Aircrafts, imagenet}'))


    i = args.start_epoch
    epochs = args.epochs
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)

        # create model
        if args.representation == 'GAvP':
            representation = {'function': GAvP,
                              'input_dim': 2048}
        elif args.representation == 'MPNCOV':
            representation = {'function': MPNCOV,
                              'iterNum': 5,
                              'input_dim': 256,
                              'dimension_reduction': None,
                              'dropout_p': None}
        elif args.representation == 'BCNN':
            representation = {'function': BCNN,
                              'input_dim': 512}
        elif args.representation == 'CBP':
            representation = {'function': CompactBilinearLayer,
                              'input_dim': 512,
                              'proj_dim': 8192}
        else:
            warnings.warn('=> You did not choose a global image representation method!')
            representation = None  # which for original vgg or alexnet

        model = get_model(args.arch,
                          representation,
                          args.num_classes,
                          args.freezed_layer,
                          pretrained=args.pretrained)

        checkpoint_prefix = os.path.join(args.exp_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(model=model)
        if args.start_epoch != 0:
            checkpoint.restore(tf.train.latest_checkpoint(args.exp_dir))

        trainer = Train(epochs=epochs, enable_function=True, model=model, batch_size=args.batchSize, strategy=strategy)

        val_ds = dataset(is_training=False, data_dir=args.dataset, pretrained=args.pretrained, arch=args.arch)\
            .make_source_dataset(batchsize=args.batchSize)
        val_ds = strategy.experimental_distribute_dataset(val_ds)

        for epoch in range(i, epochs):
            train_ds = dataset(is_training=True, data_dir=args.dataset, pretrained=args.pretrained, arch=args.arch)\
                .make_source_dataset(batchsize=args.batchSize)
            train_ds = strategy.experimental_distribute_dataset(train_ds)

            train_loss, train_top1, train_top5, val_loss, val_top1, val_top5 = \
                trainer.custom_loop(epoch, optimizer, train_ds, val_ds, strategy)

            results['train']['loss'].append(train_loss)
            results['train']['top1'].append(train_top1)
            results['train']['top5'].append(train_top5)
            results['val']['loss'].append(val_loss)
            results['val']['top1'].append(val_top1)
            results['val']['top5'].append(val_top5)

            # remember best prec@1 and save checkpoint
            checkpoint.save(file_prefix=checkpoint_prefix)
            if best_acc < val_top1:
                best_acc = val_top1
                best_epoch = int(epoch+1)
                sio.savemat(os.path.join(args.exp_dir, 'best_model.mat'),
                            {'epoch': best_epoch,
                             'params': model.get_weights()})

            sio.savemat(os.path.join(args.exp_dir, 'stats.mat'),
                        {'train_loss': results['train']['loss'],
                         'train_top1': results['train']['top1'], 'train_top5': results['train']['top5'],
                         'val_loss': results['val']['loss'],
                         'val_top1': results['val']['top1'], 'val_top5': results['val']['top5']})
            plot_curve(results, args.exp_dir, True)
            print('Epoch: {}\tTrain Loss: {:.4f}\tTop1_Accuracy: {:.4f}\tTop5_Accuracy: {:.4f}\t'
                  'Val Loss: {:.4f}\tTop1_Accuracy: {:.4f}\t Top5_Accuracy: {:.4f}\n'
                  '==============================================>BestEpoch:{}\tBest_acc:{:.4f}'.format(
                epoch+1, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5, best_epoch, best_acc))



if __name__ == '__main__':
    main()
