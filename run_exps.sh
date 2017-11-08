#!/bin/bash

# Assumes your theanorc is set properly.
python bagging_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 1 \
  --max-epochs 20 \
  --early-stopping-epochs -1
