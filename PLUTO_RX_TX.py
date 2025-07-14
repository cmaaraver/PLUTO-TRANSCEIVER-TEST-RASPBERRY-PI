import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
                            QLineEdit, QMessageBox, QProgressBar, QTextEdit,
                            QCheckBox, QFrame, QStatusBar)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QDateTime
from PyQt6.QtGui import QPalette, QColor, QFont
import pyqtgraph as pg
import logging
from datetime import datetime
import adi
import threading
import time
import usb.core
import usb.util
import os
import getpass

# Configuración global
CURRENT_USER = getpass.getuser()
DEFAULT_SAMPLE_RATE = 2083334  # Tasa de muestreo segura para PlutoSDR

class SDRException(Exception):
    """Excepción personalizada para errores del SDR"""
    pass

class PlutoSDRWorker(QThread):
    signal_metrics = pyqtSignal(float, float, float)  # BER, SNR, PER
    signal_constellation = pyqtSignal(np.ndarray)
    signal_error = pyqtSignal(str)
    signal_status = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = self._validate_config(config)
        self.running = False
        self.sdr = None
        self._setup_logging()

    def _validate_config(self, config):
        """Valida y ajusta los valores de configuración"""
        validated = config.copy()
        
        # Validación de frecuencia (70 MHz a 6 GHz)
        center_freq = int(config['center_freq'] * 1e6)  # Convertir MHz a Hz
        if not (70e6 <= center_freq <= 6e9):
            raise ValueError(f"Frecuencia {center_freq/1e6} MHz fuera de rango (70-6000 MHz)")
        validated['center_freq'] = center_freq
        
        # Validación de ancho de banda (200 kHz a 20 MHz)
        bandwidth = int(config['bandwidth'] * 1e6)  # Convertir MHz a Hz
        if not (0.2e6 <= bandwidth <= 20e6):
            raise ValueError(f"Ancho de banda {bandwidth/1e6} MHz fuera de rango (0.2-20 MHz)")
        validated['bandwidth'] = bandwidth
        
        # Validación de ganancias
        validated['tx_gain'] = min(max(int(config['tx_gain']), -90), 0)
        validated['rx_gain'] = min(max(int(config['rx_gain']), 0), 70)
        
        # Sample rate fijo para estabilidad
        validated['sample_rate'] = DEFAULT_SAMPLE_RATE
        
        return validated

    def _setup_logging(self):
        self.logger = logging.getLogger(f'PlutoSDR_{CURRENT_USER}')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            log_file = f'plutosdr_{CURRENT_USER}.log'
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def run(self):
        try:
            self._initialize_sdr()
            self.running = True
            while self.running:
                self._process_data()
                self.msleep(50)  # 20 Hz update rate
        except Exception as e:
            self.logger.error(f"Error en ejecución: {str(e)}")
            self.signal_error.emit(str(e))
        finally:
            self._cleanup()

    def _initialize_sdr(self):
        try:
            # Intentar conexiones en orden
            uris = ["ip:192.168.2.1", "usb:0"]
            connected = False
            
            for uri in uris:
                try:
                    self.sdr = adi.Pluto(uri=uri)
                    connected = True
                    self.logger.info(f"Conectado vía {uri}")
                    break
                except Exception as e:
                    self.logger.debug(f"Falló conexión {uri}: {str(e)}")
            
            if not connected:
                raise SDRException("No se pudo establecer conexión con el PlutoSDR")
            
            self._configure_device()
            self.signal_status.emit("PlutoSDR inicializado correctamente")
            
        except Exception as e:
            raise SDRException(f"Error al inicializar PlutoSDR: {str(e)}")

    def _configure_device(self):
        """Configura los parámetros del dispositivo"""
        try:
            # Configuración básica
            self.sdr.sample_rate = self.config['sample_rate']
            self.sdr.rx_rf_bandwidth = self.config['bandwidth']
            self.sdr.tx_rf_bandwidth = self.config['bandwidth']
            self.sdr.rx_lo = self.config['center_freq']
            self.sdr.tx_lo = self.config['center_freq']
            self.sdr.tx_hardwaregain_chan0 = self.config['tx_gain']
            self.sdr.rx_hardwaregain_chan0 = self.config['rx_gain']
            
            # Configuración de buffer
            self.sdr.rx_buffer_size = 1024 * 4
            self.sdr.tx_buffer_size = 1024 * 4
            self.sdr.tx_cyclic_buffer = False
            
            # Verificar configuración
            self._verify_configuration()
            
        except Exception as e:
            raise SDRException(f"Error en configuración: {str(e)}")

    def _verify_configuration(self):
        """Verifica la configuración aplicada"""
        tolerance = 1e-6
        checks = [
            (self.sdr.sample_rate, self.config['sample_rate'], "sample_rate"),
            (self.sdr.rx_rf_bandwidth, self.config['bandwidth'], "bandwidth"),
            (self.sdr.rx_lo, self.config['center_freq'], "frequency")
        ]
        
        for actual, expected, param in checks:
            if abs(actual - expected) > expected * tolerance:
                raise SDRException(
                    f"Error de configuración en {param}: "
                    f"esperado {expected}, actual {actual}"
                )

    def _process_data(self):
        """Procesa datos del SDR"""
        try:
            # Generar datos de prueba
            test_bits = np.random.randint(0, 2, 1024)
            
            # Modular y transmitir
            symbols = self._qpsk_modulate(test_bits)
            self.sdr.tx(symbols)
            
            # Recibir y procesar
            rx_samples = self.sdr.rx()
            rx_bits = self._qpsk_demodulate(rx_samples)
            
            # Calcular métricas
            ber = self._calculate_ber(test_bits[:len(rx_bits)], rx_bits)
            snr = self._calculate_snr(rx_samples)
            per = self._calculate_per(test_bits[:len(rx_bits)], rx_bits)
            
            # Emitir resultados
            self.signal_metrics.emit(ber, snr, per)
            self.signal_constellation.emit(rx_samples)
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento: {str(e)}")
            self.signal_error.emit(f"Error en procesamiento: {str(e)}")

    def _qpsk_modulate(self, bits):
        """Modulación QPSK optimizada"""
        symbols = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
        bit_pairs = bits.reshape((-1, 2))
        indices = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]
        return symbols[indices]

    def _qpsk_demodulate(self, symbols):
        """Demodulación QPSK optimizada"""
        return np.column_stack([
            symbols.real >= 0,
            symbols.imag >= 0
        ]).flatten()

    def _calculate_ber(self, sent, received):
        """Calcula Bit Error Rate"""
        if len(sent) != len(received):
            return 1.0
        return np.mean(sent != received)

    def _calculate_snr(self, signal):
        """Calcula Signal-to-Noise Ratio"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.var(np.abs(signal))
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    def _calculate_per(self, sent, received):
        """Calcula Packet Error Rate"""
        packet_size = 8
        if len(sent) != len(received):
            return 1.0
        packets_sent = sent.reshape(-1, packet_size)
        packets_received = received.reshape(-1, packet_size)
        return np.mean(np.any(packets_sent != packets_received, axis=1))

    def stop(self):
        """Detiene el worker de forma segura"""
        self.running = False
        self.wait()

    def _cleanup(self):
        """Limpia recursos"""
        if self.sdr:
            try:
                self.sdr.tx_destroy_buffer()
                self.sdr.rx_destroy_buffer()
                self.sdr.close()
            except Exception as e:
                self.logger.error(f"Error en cleanup: {str(e)}")

class ConstellationWidget(pg.PlotWidget):
    """Widget para visualización de constelación"""
    def __init__(self):
        super().__init__()
        self._setup_plot()

    def _setup_plot(self):
        self.setBackground('k')
        self.showGrid(x=True, y=True)
        self.setLabel('left', 'Q')
        self.setLabel('bottom', 'I')
        self.getPlotItem().setRange(xRange=(-2, 2), yRange=(-2, 2))
        
        # Puntos de constelación recibidos
        self.scatter = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush('g')
        )
        self.addItem(self.scatter)
        
        # Puntos de constelación ideales
        ideal_points = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
        self.ideal_scatter = pg.ScatterPlotItem(
            pos=np.column_stack((ideal_points.real, ideal_points.imag)),
            size=10,
            pen=pg.mkPen('r'),
            brush=pg.mkBrush(None)
        )
        self.addItem(self.ideal_scatter)

    def update_constellation(self, samples):
        self.scatter.setData(pos=np.column_stack((samples.real, samples.imag)))

class MetricsDisplay(QFrame):
    """Widget para mostrar métricas"""
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Crear displays para métricas
        self.ber_label = QLabel("BER: 0.000")
        self.snr_label = QLabel("SNR: 0.00 dB")
        self.per_label = QLabel("PER: 0.000")
        
        # Estilo de las etiquetas
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        for label in [self.ber_label, self.snr_label, self.per_label]:
            label.setFont(font)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        
        self.setLayout(layout)

    def update_metrics(self, ber, snr, per):
        """Actualiza las métricas mostradas"""
        self.ber_label.setText(f"BER: {ber:.6f}")
        self.snr_label.setText(f"SNR: {snr:.2f} dB")
        self.per_label.setText(f"PER: {per:.6f}")
        
        # Actualizar colores según valores
        self._update_color(self.ber_label, ber, 0.1)
        self._update_color(self.snr_label, snr, 10, inverse=True)
        self._update_color(self.per_label, per, 0.1)

    def _update_color(self, label, value, threshold, inverse=False):
        """Actualiza el color según el valor"""
        if inverse:
            good = value > threshold
        else:
            good = value < threshold
        label.setStyleSheet("color: green" if good else "color: red")

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PLUTO SDR TRANSCEIVER TEST - {CURRENT_USER}")
        self.setMinimumSize(1200, 800)
        self._setup_ui()
        self._setup_logging()
        
        # Mostrar información inicial
        self.statusBar().showMessage(
            f"Usuario: {CURRENT_USER} | "
            f"Fecha: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _setup_logging(self):
        self.logger = logging.getLogger(f'MainWindow_{CURRENT_USER}')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            log_file = f'plutosdr_gui_{CURRENT_USER}.log'
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Paneles
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        visualization_panel = self._create_visualization_panel()
        main_layout.addWidget(visualization_panel, 2)
        
        # Barra de estado
        self.statusBar().showMessage("Listo")
        
        # Estilo
        self._setup_style()

    def _setup_style(self):
        """Configura el estilo de la aplicación"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def _create_control_panel(self):
        """Crea el panel de control"""
        control_group = QGroupBox("Panel de Control")
        layout = QVBoxLayout()

        # Configuración de frecuencia
        freq_group = self._create_frequency_group()
        layout.addWidget(freq_group)

        # Configuración de ganancia
        gain_group = self._create_gain_group()
        layout.addWidget(gain_group)

        # Botones de control
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_acquisition)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Detener")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)

        # Log de estado
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        layout.addWidget(QLabel("Log de Estado:"))
        layout.addWidget(self.status_log)

        control_group.setLayout(layout)
        return control_group

    def _create_frequency_group(self):
        """Crea el grupo de configuración de frecuencia"""
        freq_group = QGroupBox("Configuración de Frecuencia")
        layout = QVBoxLayout()

        # Frecuencia central (MHz)
        self.center_freq = QDoubleSpinBox()
        self.center_freq.setRange(70, 6000)
        self.center_freq.setValue(435)
        self.center_freq.setSuffix(" MHz")
        self.center_freq.setDecimals(3)
        layout.addWidget(QLabel("Frecuencia Central:"))
        layout.addWidget(self.center_freq)

        # Ancho de banda (MHz)
        self.bandwidth = QDoubleSpinBox()
        self.bandwidth.setRange(0.2, 20)
        self.bandwidth.setValue(0.5)
        self.bandwidth.setSuffix(" MHz")
        self.bandwidth.setDecimals(3)
        layout.addWidget(QLabel("Ancho de Banda:"))
        layout.addWidget(self.bandwidth)

        freq_group.setLayout(layout)
        return freq_group

    def _create_gain_group(self):
        """Crea el grupo de configuración de ganancia"""
        gain_group = QGroupBox("Configuración de Ganancia")
        layout = QVBoxLayout()

        # Ganancia TX
        self.tx_gain = QSpinBox()
        self.tx_gain.setRange(-90, 0)
        self.tx_gain.setValue(-20)
        self.tx_gain.setSuffix(" dB")
        layout.addWidget(QLabel("Ganancia TX:"))
        layout.addWidget(self.tx_gain)

        # Ganancia RX
        self.rx_gain = QSpinBox()
        self.rx_gain.setRange(0, 70)
        self.rx_gain.setValue(30)
        self.rx_gain.setSuffix(" dB")
        layout.addWidget(QLabel("Ganancia RX:"))
        layout.addWidget(self.rx_gain)

        gain_group.setLayout(layout)
        return gain_group

    def _create_visualization_panel(self):
        """Crea el panel de visualización"""
        visualization_group = QGroupBox("Visualización")
        layout = QVBoxLayout()

        # Diagrama de constelación
        self.constellation = ConstellationWidget()
        layout.addWidget(self.constellation)

        # Display de métricas
        self.metrics_display = MetricsDisplay()
        layout.addWidget(self.metrics_display)

        visualization_group.setLayout(layout)
        return visualization_group

    def start_acquisition(self):
        """Inicia la adquisición de datos"""
        try:
            config = {
                'center_freq': self.center_freq.value(),  # En MHz
                'bandwidth': self.bandwidth.value(),      # En MHz
                'tx_gain': self.tx_gain.value(),
                'rx_gain': self.rx_gain.value(),
                'sample_rate': DEFAULT_SAMPLE_RATE
            }

            self.worker = PlutoSDRWorker(config)
            self.worker.signal_metrics.connect(self.update_metrics)
            self.worker.signal_constellation.connect(self.update_constellation)
            self.worker.signal_error.connect(self.handle_error)
            self.worker.signal_status.connect(self.update_status)
            self.worker.start()

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_status("Adquisición iniciada")

        except Exception as e:
            self.handle_error(str(e))

    def stop_acquisition(self):
        """Detiene la adquisición de datos"""
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Adquisición detenida")

    def update_metrics(self, ber, snr, per):
        """Actualiza las métricas mostradas"""
        self.metrics_display.update_metrics(ber, snr, per)

    def update_constellation(self, samples):
        """Actualiza el diagrama de constelación"""
        self.constellation.update_constellation(samples)

    def handle_error(self, error_msg):
        """Maneja los errores de la aplicación"""
        QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"ERROR: {error_msg}")
        self.stop_acquisition()

    def update_status(self, message):
        """Actualiza el log de estado"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.status_log.append(f"{timestamp}: {message}")
        self.statusBar().showMessage(f"Usuario: {CURRENT_USER} | Último evento: {message}")

    def closeEvent(self, event):
        """Maneja el cierre de la aplicación"""
        self.stop_acquisition()
        event.accept()

def main():
    """Función principal"""
    try:
        app = QApplication(sys.argv)
        
        # Verificar dependencias
        required_packages = ['PyQt6', 'numpy', 'adi', 'pyqtgraph']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                QMessageBox.critical(None, "Error",
                    f"Falta el paquete {package}. Por favor, instálelo con:\n"
                    f"pip install {package}")
                return 1
        
        # Configurar logging global
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'plutosdr_app_{CURRENT_USER}.log'),
                logging.StreamHandler()
            ]
        )
        
        window = MainWindow()
        window.show()
        return app.exec()
    
    except Exception as e:
        logging.error(f"Error en la aplicación: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())