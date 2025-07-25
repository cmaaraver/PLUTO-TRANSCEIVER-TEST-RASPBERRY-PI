# PlutoSDR BER Test Application

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PlutoSDR](https://img.shields.io/badge/Hardware-PlutoSDR-orange.svg)](https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html)

Una aplicación completa en Python para realizar pruebas de **BER (Bit Error Rate)** utilizando dos **PlutoSDR** con transmisión digital en banda ISM. Incluye visualización en tiempo real de SNR, constelación QPSK, espectro RF y estadísticas de transmisión.

![Demo Screenshot](docs/demo_screenshot.png)

## 🎯 Características Principales

- 📡 **Transmisión RF digital** entre dos PlutoSDR
- 🔢 **Modulación QPSK** optimizada para 25 kHz de ancho de banda
- 📊 **Visualización en tiempo real** de métricas RF
- 🧮 **Cálculo automático de BER** y comparación de archivos
- 🎲 **Generación de datos aleatorios** para pruebas
- 📈 **Gráficas dinámicas** de SNR, constelación y espectro
- ⚙️ **Configuración flexible** de parámetros RF

## 🛠️ Configuración Técnica

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Frecuencia** | 433.92 MHz | Banda ISM libre |
| **Ancho de banda** | 25 kHz | Canal estrecho optimizado |
| **Modulación** | QPSK | 2 bits por símbolo |
| **Sample Rate** | 250 kHz | 10x oversampling |
| **Filtro** | Root Raised Cosine | α = 0.35 |
| **Conexión** | USB | 192.168.2.1 (por defecto) |

## 🚀 Instalación

### Prerrequisitos

- Python 3.7 o superior
- Dos PlutoSDR (uno para TX, otro para RX)
- Drivers PlutoSDR instalados

### Dependencias

```bash
pip install pyadi-iio numpy matplotlib
```

### Instalación de Drivers PlutoSDR

**Linux/macOS:**
```bash
# Instalar libiio
sudo apt-get install libiio-utils  # Ubuntu/Debian
brew install libiio                # macOS

# Verificar conexión
iio_info -s
```

**Windows:**
- Descargar drivers desde [Analog Devices](https://github.com/analogdevicesinc/plutosdr-fw/releases)
- Instalar IIO Oscilloscope

## 📖 Uso de la Aplicación

### Ejecución Rápida (Interactiva)

```bash
python pluto_ber_ism.py
```

La aplicación te preguntará si quieres configurar el PlutoSDR como transmisor (TX) o receptor (RX).

### Modo Transmisor

```bash
# Transmisión única
python pluto_ber_ism.py --mode tx --file datos_test.bin

# Transmisión continua cada 3 segundos
python pluto_ber_ism.py --mode tx --continuous --interval 3.0

# Ajustar potencia de transmisión
python pluto_ber_ism.py --mode tx --tx-power -5 --continuous
```

### Modo Receptor

```bash
# Recepción durante 2 minutos
python pluto_ber_ism.py --mode rx --duration 120

# Ajustar ganancia de recepción
python pluto_ber_ism.py --mode rx --rx-gain 45 --duration 60

# Especificar archivo de referencia para comparación
python pluto_ber_ism.py --mode rx --file datos_test.bin --duration 180
```

## ⚙️ Parámetros de Configuración

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--mode` | Modo de operación: `tx` o `rx` | Interactivo |
| `--freq` | Frecuencia central (Hz) | 433.92e6 |
| `--sample-rate` | Tasa de muestreo (Hz) | 250000 |
| `--file` | Archivo de datos | datos_test.bin |
| `--duration` | Duración recepción (s) | 60 |
| `--continuous` | Transmisión continua | False |
| `--interval` | Intervalo entre TX (s) | 2.0 |
| `--tx-power` | Potencia TX (dBm) | -10 |
| `--rx-gain` | Ganancia RX (dB) | 50 |

## 📊 Métricas y Visualización

### Gráficas en Tiempo Real

1. **SNR Temporal**: Evolución de la relación señal/ruido
2. **Espectro RF**: Análisis frecuencial de la señal recibida
3. **Constelación QPSK**: Calidad de la modulación digital
4. **Panel de Estadísticas**: Métricas en tiempo real

### Métricas Calculadas

- **BER (Bit Error Rate)**: Tasa de error de bits
- **SNR (Signal-to-Noise Ratio)**: Relación señal/ruido
- **Pérdida de paquetes**: Porcentaje de paquetes perdidos
- **Tasa de éxito**: Porcentaje de transmisión exitosa
- **Comparación bit a bit**: Entre archivo original y recibido

## 📁 Archivos Generados

```
datos_test.bin              # Archivo original (binario)
datos_test_hex.txt          # Versión hexadecimal legible
datos_test_received.bin     # Datos recibidos
datos_test_received_hex.txt # Datos recibidos en hexadecimal
```

## 🔧 Configuración de Hardware

### Conexión Básica

```
Ordenador 1 (TX)    Ordenador 2 (RX)
    |                    |
    USB                  USB
    |                    |
PlutoSDR (TX) ~~~RF~~~ PlutoSDR (RX)
```

### Configuración de Red

- **Conexión USB**: Cada PlutoSDR se conecta a su PC por USB
- **IP por defecto**: 192.168.2.1 (configuración automática)
- **Transmisión**: A través de RF en 433.92 MHz

### Antenas Recomendadas

- **Frecuencia**: 433 MHz
- **Tipo**: Dipolo λ/2 ≈ 17.3 cm
- **Conector**: SMA (según modelo PlutoSDR)

## 📖 Ejemplos de Uso

### Ejemplo 1: Prueba Básica de BER

```bash
# Terminal 1 (Transmisor)
python pluto_ber_ism.py --mode tx --continuous --interval 2

# Terminal 2 (Receptor) 
python pluto_ber_ism.py --mode rx --duration 300
```

### Ejemplo 2: Prueba con Potencia Reducida

```bash
# Transmisor con baja potencia
python pluto_ber_ism.py --mode tx --tx-power -20 --continuous

# Receptor con alta ganancia
python pluto_ber_ism.py --mode rx --rx-gain 60 --duration 120
```

### Ejemplo 3: Análisis de Frecuencia Personalizada

```bash
# Usar frecuencia ISM alternativa (868 MHz Europa)
python pluto_ber_ism.py --mode tx --freq 868e6 --continuous
python pluto_ber_ism.py --mode rx --freq 868e6 --duration 180
```

## 🐛 Solución de Problemas

### PlutoSDR No Detectado

```bash
# Verificar conexión
iio_info -s

# Debería mostrar: ip:192.168.2.1
```

### Error de Conexión

```python
# Verificar que PlutoSDR esté conectado
import adi
sdr = adi.Pluto("ip:192.168.2.1")
print("Conexión exitosa")
```

### Problemas de RF

- **Verificar antenas**: Conectadas correctamente
- **Distancia**: Comenzar con ~1 metro entre PlutoSDRs
- **Interferencia**: Alejarse de WiFi 2.4 GHz
- **Potencia**: Ajustar `--tx-power` y `--rx-gain`

### Dependencias Faltantes

```bash
# Reinstalar dependencias
pip uninstall pyadi-iio
pip install pyadi-iio --upgrade

# Verificar numpy y matplotlib
pip install numpy matplotlib --upgrade
```

## 📚 Documentación Técnica

### Arquitectura del Sistema

```
[Archivo de bits] → [Modulador QPSK] → [PlutoSDR TX] 
                                            |
                                           RF
                                            |
[Archivo recibido] ← [Demodulador QPSK] ← [PlutoSDR RX]
```

### Protocolo de Paquetes

```
[Preámbulo 72bits] [Longitud 16bits] [Datos Nbits] [Checksum 8bits]
```

### Parámetros QPSK

- **Constelación**: {1+1j, -1+1j, -1-1j, 1-1j} / √2
- **Símbolos por segundo**: 12.5 kHz
- **Muestras por símbolo**: 8
- **Roll-off factor**: α = 0.35

### Ideas para Contribuir

- [ ] Soporte para otras modulaciones (8PSK, QAM)
- [ ] Interfaz gráfica (GUI)
- [ ] Exportación de datos (CSV, JSON)
- [ ] Modo automatizado de pruebas
- [ ] Soporte para múltiples frecuencias
- [ ] Análisis de canal (fading, multipath)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

