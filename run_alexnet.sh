#!/bin/bash
source /home/xiaohui/.bashrc

. /root/xhzhao/distro/install/bin/torch-activate

 th main_alexnet.lua -data  /root/bingyan/imagenet/
