#!/bin/bash

# Assumes your theanorc is set properly.
python evaluate_bagging_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -e ./experiments/1510174775-9449394
