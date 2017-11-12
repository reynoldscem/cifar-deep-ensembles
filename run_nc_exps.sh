#!/bin/bash

# Assumes your theanorc is set properly.
python nc_learning_experiment.py \
  -d ../cifar/cifar-10-batches-py \
  -k 2 \
  --max-epochs 50 \
  --early-stopping-epochs 4 \
  --base-power 4
