#!/bin/bash

# rm -rf ./experiments
# Assumes your theanorc is set properly.
python bagging_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 128 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 4

python bagging_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 128 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 5
