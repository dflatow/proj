# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet_final import *
import cPickle as pickle
import os
from cs231n.data_utils import get_CIFAR10_data
from argparse import ArgumentParser
import logging

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-18, np.abs(x) + np.abs(y))))


def permute_lables(p, labels_orig, classes=9): 
    labels = labels_orig.copy()
    n_labels = len(labels)
    n_to_permute = int(np.floor(p * n_labels))
    inds_to_permute = np.random.choice(n_labels, n_to_permute, replace=False)
    new_labels = np.random.choice(classes, n_to_permute, replace=True)
    labels[inds_to_permute] = new_labels
    return labels



if __name__ == "__main__":


  parser = ArgumentParser()
  parser.add_argument("p", nargs="?", default=0.0, type=float)
  parser.add_argument("--update", default="momentum", type=str)
  parser.add_argument("--reg", default=0.00008, type=float)
  parser.add_argument("--momentum", default=0.9, type=float)
  parser.add_argument("--learning_rate", default=0.0014, type=float)
  parser.add_argument("--batch_size", default=300, type=int)
  parser.add_argument("--num_epochs", default=30, type=int)  
  parser.add_argument("--data_dir", default="/data/cnn_proj/code/cs231n/datasets/cifar-10-batches-py", type=str)
  args = parser.parse_args()

  # unpack
  p = args.p
  update = args.update
  reg = args.reg
  momentum = args.momentum
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  num_epochs = args.num_epochs
  data_dir = args.data_dir
  
  add_to_suffix = lambda name, val: "_" + name + "=" + str(val) + "_"
  file_suffix = ""
  file_suffix += add_to_suffix("p", p)
  file_suffix += add_to_suffix("update", update)
  file_suffix += add_to_suffix("momentum", momentum)  
  file_suffix += add_to_suffix("reg", reg)
  file_suffix += add_to_suffix("learning_rate", learning_rate)
  file_suffix += add_to_suffix("batch_size", batch_size)
  file_suffix += add_to_suffix("num_epochs", num_epochs)      

  log_file = "log" + file_suffix + ".log"
  logging.basicConfig(filename=log_file,level=logging.DEBUG, 
                    format='%(levelname)s %(asctime)s %(message)s')

  logging.debug('START')

  
  X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(data_dir=data_dir)
  y_train_permuted = permute_lables(p, y_train)
  model = init_three_layer_convnet(filter_size=5, weight_scale=5e-3, num_filters=32)
  trainer = ClassifierTrainer()
  best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          X_train[:300], y_train_permuted[:300], X_val, y_val, model, three_layer_convnet, update=update,
          reg=reg, momentum=momentum, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
          verbose=True, logging=logging)
  
  model_fname = "model" + file_suffix + ".p"
  with open(model_fname, 'w+') as f:
       pickle.dump(model, f)

  scores_test = three_layer_convnet(X_test.transpose(0, 3, 1, 2), model)
  logging.debug('Test accuracy: %f' % (np.mean(np.argmax(scores_test, axis=1) == y_test),))
  logging.debug('DONE')
