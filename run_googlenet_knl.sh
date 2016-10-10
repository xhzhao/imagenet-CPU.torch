#!/bin/bash
source /home/wenting/.bashrc


KMP_AFFINITY=scatter granularity=fine OMP_NUM_THREADS=68 th main_googlenet_knl.lua -data  /data/imagenet/ilsvrc2012/
