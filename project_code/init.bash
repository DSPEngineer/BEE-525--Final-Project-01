#! /usr/bin/bash

sudo apt update
sudo apt install -y \
    python-is-python3 \
    python3.11-venv \
    libcap-dev \
    python3-picamzero

    #sudo apt install -y python3-opencv

## The follwing installs non-PIP versions - they don't seem to work
#sudo apt install -y python3-picamera2
#sudo apt reinstall -y python3-picamzero
#sudo apt install -y python3-opencv

## Prerequisites for TensorFlow
#sudo apt install -y libatlas-base-dev libhdf5-dev python3-h5py
#pip install tensorflow --break-system-packages
#sudo apt purge  -y python3-numpy python3-matplotlib

#echo "-- -- -- -- -- -- -- -- -- -- Environment -- -- -- -- -- -- -- -- -- --"
python3 -m venv env

source  env/bin/activate

echo "-- -- -- -- -- -- -- -- -- -- UNINSTALL -- -- -- -- -- -- -- -- -- --"
pip uninstall -y --break-system-packages \
    tensorflow \
    python-dateutil

pip uninstall -y \
    opencv-python \
    image \
    picamzero \
    numpy \
    matplotlib pillow

echo "-- -- -- -- -- -- -- -- -- --  INSTALL  -- -- -- -- -- -- -- -- -- --"
#pip install \
#    numpy==1.26.4 \
#    opencv-python \
#    pillow  \
#    matplotlib

#pip install  --break-system-packages \
#    tensorflow

#pip install  python-dateutil --break-system-packages

#pip install  tensorflow  --break-system-packages

#pip install  picamzero
#pip install  pillow  matplotlib

