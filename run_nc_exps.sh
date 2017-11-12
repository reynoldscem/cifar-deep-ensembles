#!/bin/bash

# Assumes your theanorc is set properly.
python nc_learning_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 4 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 4
python nc_learning_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 8 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 4
python nc_learning_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 12 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 4
