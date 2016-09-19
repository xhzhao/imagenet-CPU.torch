#!/bin/bash
source /home/xiaohui/.bashrc


OMP_NUM_THREADS=44 th main_googlenet.lua -data  /data/imagenet/ilsvrc2012/
