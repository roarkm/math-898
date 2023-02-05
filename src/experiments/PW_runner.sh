#!/usr/bin/env bash

for d in $(seq 2 2 10); do
  echo $d
  for i in $(seq 1 100); do
    echo run $i;
    python src/experiments/PW_run_times.py -d $d;
  done
done
