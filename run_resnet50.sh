#!/bin/bash
source /home/wenting/.bashrc


  OMP_NUM_THREADS=44 th main_resnet50.lua -data  /data/imagenet/ilsvrc2012/
