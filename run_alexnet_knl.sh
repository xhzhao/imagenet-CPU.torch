#!/bin/bash
source /home/xiaohui/.bashrc


KMP_AFFINITY=scatter granularity=fine OMP_NUM_THREADS=68  th main_alexnet.lua -data  /data/imagenet/ilsvrc2012/
