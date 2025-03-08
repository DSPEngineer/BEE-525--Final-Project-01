#! /usr/bin/bash

sudo apt update
sudo apt install -y python-is-python3
sudo apt install -y python3.11-venv
sudo apt install -y libcap-dev
#sudo apt install -y python3-opencv

## The follwing installs non-PIP versions - they don't seem to work
#sudo apt install -y python3-picamera2
#sudo apt install -y python3-picamzero
#sudo apt install -y python3-opencv
## Prerequisites for TensorFlow
#sudo apt install -y libatlas-base-dev libhdf5-dev python3-h5py
#sudo apt install -y python3-numpy
#sudo apt install -y python3-matplotlib
#pip install tensorflow --break-system-packages

echo "-- -- -- -- -- -- -- -- -- -- Environment -- -- -- -- -- -- -- -- -- --"
python3 -m venv env

. env/bin/activate

echo "-- -- -- -- -- -- -- -- -- -- UNINSTALL -- -- -- -- -- -- -- -- -- --"
pip uninstall -y picamzero
pip uninstall -y numpy
pip uninstall -y matplotlib
pip uninstall -y tensorflow  --break-system-packages

echo "-- -- -- -- -- -- -- -- -- --  INSTALL  -- -- -- -- -- -- -- -- -- --"
pip install  numpy==1.26.4
pip install  matplotlib
pip install  tensorflow  --break-system-packages
pip install  picamzero



