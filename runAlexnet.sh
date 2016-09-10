#!/bin/bash
source /home/xiaohui/.bashrc


  OMP_NUM_THREADS=44 th main_alexnet.lua -data  /data/imagenet/ilsvrc2012/
