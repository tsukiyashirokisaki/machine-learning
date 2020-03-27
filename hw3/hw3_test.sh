#!/bin/bash
wget https://www.dropbox.com/s/xt8g5vyyq7syzm7/model.pth?dl=1 -O model.pth
python3 test.py $1 $2 