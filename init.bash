#! /usr/bin/bash

#export PYTHONPATH=/usr/lib/python3/dist-packages/:~/.local/lib/python3.11/site-packages/
sudp apt install python3-picamzero

python3 -m venv env

source env/bin/activate
pip3 install tensorflow
pip3 install opencv-python
pip3 install numpy==1.26.4
pip3 install matplotlib
pip3 install keyboard



