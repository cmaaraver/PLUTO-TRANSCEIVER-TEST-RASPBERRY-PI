#!/bin/bash

echo "=== PLUTO SDR TRANSCEIVER TEST - Instalador de Dependencias ==="
echo "Fecha de instalación: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Usuario: $USER"
echo "=================================================="

# Actualizar el sistema
echo "[1/5] Actualizando el sistema..."
sudo apt-get update
sudo apt-get upgrade -y

# Instalar dependencias del sistema
echo "[2/5] Instalando dependencias del sistema..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libusb-1.0-0-dev \
    libiio-dev \
    python3-libiio \
    git \
    wget \
    build-essential

# Actualizar pip
echo "[3/5] Actualizando pip..."
python3 -m pip install --upgrade pip

# Instalar dependencias de Python
echo "[4/5] Instalando dependencias de Python..."
pip3 install --upgrade \
    PyQt6 \
    numpy \
    pyadi-iio \
    pyqtgraph \
    pyusb \
    setuptools \
    wheel

# Verificar instalación
echo "[5/5] Verificando instalación..."
python3 -c "
import sys
required = ['PyQt6', 'numpy', 'adi', 'pyqtgraph', 'usb']
missing = []
for package in required:
    try:
        __import__(package)
        print(f'✓ {package} instalado correctamente')
    except ImportError:
        missing.append(package)
        print(f'✗ {package} no se pudo instalar')
if missing:
    print('\n⚠ Advertencia: Algunos paquetes no se instalaron correctamente')
    sys.exit(1)
else:
    print('\n✓ Todos los paquetes se instalaron correctamente')
"

echo "=================================================="
echo "Instalación completada!"
echo "Para ejecutar la aplicación use: python3 pluto_sdr_transceiver.py"
