#!/bin/bash
source /home/xiaohui/.bashrc

. /home/xiaohui/torch_icc/install/bin/torch-activate

 th main_googlenet.lua -data  /data/imagenet/ilsvrc2012/
