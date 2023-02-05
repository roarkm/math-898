#!/usr/bin/env bash

for d in $(seq 8 16); do
  for i in $(seq 1 100); do
    echo run $i;
    python src/experiments/run_times.py -e 0 -d $d;
    python src/experiments/run_times.py -e 1 -d $d;
  done
done
