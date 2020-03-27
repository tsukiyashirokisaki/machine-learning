#!/bin/bash
python3 BOW.py $1 $2 $3 
python3 lstm.py $1 $2 $3 
python3 gru.py $1 $2 $3 