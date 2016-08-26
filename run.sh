#!/bin/bash
source /home/xiaohui/.bashrc



KMP_AFFINITY=scatter,granularity=fine,0,1  OMP_NUM_THREADS=44 th main.lua -data  /data/imagenet/ilsvrc2012/
