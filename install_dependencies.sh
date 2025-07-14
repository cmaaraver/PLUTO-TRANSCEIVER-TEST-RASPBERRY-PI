#!/bin/bash

# Actualizar el sistema
echo "Actualizando el sistema..."
sudo apt-get update
sudo apt-get upgrade -y

# Instalar dependencias del sistema
echo "Instalando dependencias del sistema..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libusb-1.0-0-dev \
    libiio-dev \
    python3-libiio

# Actualizar pip
echo "Actualizando pip..."
python3 -m pip install --upgrade pip

# Instalar dependencias de Python
echo "Instalando dependencias de Python..."
pip3 install --upgrade \
    PyQt6 \
    numpy \
    pyadi-iio \
    pyqtgraph \
    pyusb

echo "Instalación completada!"