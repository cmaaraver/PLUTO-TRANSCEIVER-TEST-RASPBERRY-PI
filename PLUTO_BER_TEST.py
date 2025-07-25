import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
import threading
import queue
import sys
import os
from collections import deque
import struct
import hashlib
import random

try:
    import iio
    print("pyadi-iio disponible")
except ImportError:
    print("Error: Instala pyadi-iio con: pip install pyadi-iio")
    sys.exit(1)

try:
    import adi
    print("adi (PlutoSDR) disponible")
except ImportError:
    print("Error: Instala adi con: pip install pyadi-iio")
    sys.exit(1)

class DigitalModem:
    """Implementa modulación/demodulación digital QPSK optimizada para 25kHz"""
    
    def __init__(self, bandwidth=25000):
        self.bandwidth = bandwidth
        self.symbol_rate = bandwidth / 2  # 12.5 kHz symbol rate
        self.samples_per_symbol = 8
        self.constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
    def modulate_qpsk(self, bits):
        """Modula bits en QPSK con filtro para 25kHz"""
        # Agregar padding si es necesario
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        
        # Convertir pares de bits a símbolos
        symbols = []
        for i in range(0, len(bits), 2):
            bit_pair = bits[i:i+2]
            symbol_idx = bit_pair[0] * 2 + bit_pair[1]
            symbols.append(self.constellation[symbol_idx])
        
        symbols = np.array(symbols)
        
        # Pulso conformador optimizado para 25kHz
        pulse = self._root_raised_cosine_pulse(alpha=0.35)
        
        # Upsample
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol, dtype=complex)
        upsampled[::self.samples_per_symbol] = symbols
        
        # Filtrado
        modulated = np.convolve(upsampled, pulse, mode='same')
        
        # Normalizar potencia
        modulated = modulated / np.sqrt(np.mean(np.abs(modulated)**2))
        
        return modulated
    
    def demodulate_qpsk(self, received_signal):
        """Demodula señal QPSK"""
        # Matched filter
        pulse = self._root_raised_cosine_pulse(alpha=0.35)
        filtered = np.convolve(received_signal, np.conj(pulse[::-1]), mode='same')
        
        # Sincronización de símbolo (simplificada)
        power = np.abs(filtered)**2
        symbol_timing = self._find_symbol_timing(power)
        
        # Downsampling con timing correcto
        symbols = filtered[symbol_timing::self.samples_per_symbol]
        
        # Corrección de fase (simplificada)
        symbols = self._phase_correction(symbols)
        
        # Decisión de símbolos
        bits = []
        for symbol in symbols:
            distances = np.abs(self.constellation - symbol)
            closest_idx = np.argmin(distances)
            
            bit1 = closest_idx // 2
            bit0 = closest_idx % 2
            bits.extend([bit1, bit0])
        
        return np.array(bits, dtype=int)
    
    def _root_raised_cosine_pulse(self, alpha=0.35, span=8):
        """Genera pulso root raised cosine optimizado"""
        n = self.samples_per_symbol * span
        t = np.arange(-n//2, n//2+1) / self.samples_per_symbol
        
        # Root raised cosine
        pulse = np.zeros_like(t, dtype=float)
        
        for i, time in enumerate(t):
            if abs(time) < 1e-10:  # t = 0
                pulse[i] = 1 + alpha * (4/np.pi - 1)
            elif abs(abs(time) - 1/(4*alpha)) < 1e-10:  # t = ±1/(4α)
                pulse[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi) * np.sin(np.pi/(4*alpha)) + 
                           (1-2/np.pi) * np.cos(np.pi/(4*alpha)))
            else:
                numerator = np.sin(np.pi * time * (1-alpha)) + 4*alpha*time*np.cos(np.pi*time*(1+alpha))
                denominator = np.pi * time * (1 - (4*alpha*time)**2)
                pulse[i] = numerator / denominator
        
        # Normalizar
        pulse = pulse / np.sqrt(np.sum(pulse**2))
        return pulse
    
    def _find_symbol_timing(self, power_signal):
        """Encuentra el timing óptimo de símbolo"""
        # Usar autocorrelación para encontrar periodicidad
        correlation = np.correlate(power_signal, power_signal[::self.samples_per_symbol], mode='valid')
        return np.argmax(correlation[:self.samples_per_symbol])
    
    def _phase_correction(self, symbols):
        """Corrección básica de fase usando símbolos piloto"""
        if len(symbols) < 10:
            return symbols
        
        # Estimar rotación de fase usando algunos símbolos
        phase_estimates = []
        for i in range(min(10, len(symbols))):
            symbol = symbols[i]
            closest_constellation = self.constellation[np.argmin(np.abs(self.constellation - symbol))]
            if abs(closest_constellation) > 0.1:
                phase_error = np.angle(closest_constellation) - np.angle(symbol)
                phase_estimates.append(phase_error)
        
        if phase_estimates:
            avg_phase_error = np.mean(phase_estimates)
            correction = np.exp(1j * avg_phase_error)
            return symbols * correction
        
        return symbols

class PlutoSDRTransmitter:
    """Clase para transmisión RF con PlutoSDR en banda ISM"""
    
    def __init__(self, center_freq=433.92e6, sample_rate=250000):  # Banda ISM 433 MHz
        self.center_freq = int(center_freq)
        self.sample_rate = int(sample_rate)  # 250 kHz para acomodar 25kHz bandwidth
        self.bandwidth = 25000
        self.modem = DigitalModem(self.bandwidth)
        
        try:
            # Conexión USB por defecto
            self.sdr = adi.Pluto("ip:192.168.2.1")
            
            # Configuración TX
            self.sdr.tx_rf_bandwidth = int(self.sample_rate)
            self.sdr.tx_lo = int(self.center_freq)
            self.sdr.tx_hardwaregain_chan0 = -10  # dBm (ajustable)
            self.sdr.sample_rate = int(self.sample_rate)
            
            print(f"PlutoSDR TX configurado:")
            print(f"  Frecuencia: {self.center_freq/1e6:.3f} MHz (Banda ISM)")
            print(f"  Sample Rate: {self.sample_rate/1000:.0f} kHz")
            print(f"  Bandwidth: {self.bandwidth/1000:.0f} kHz")
            print(f"  Potencia TX: -10 dBm")
            
        except Exception as e:
            print(f"Error conectando PlutoSDR TX: {e}")
            print("Asegúrate de que PlutoSDR esté conectado por USB")
            sys.exit(1)
    
    def generate_random_bits_file(self, filename, num_bytes=1000):
        """Genera archivo con bits aleatorios"""
        try:
            # Generar bytes aleatorios
            random_bytes = bytes([random.randint(0, 255) for _ in range(num_bytes)])
            
            # Guardar como archivo binario y también como texto hexadecimal
            with open(filename, 'wb') as f:
                f.write(random_bytes)
            
            # Crear versión texto para visualización
            hex_filename = filename.replace('.bin', '_hex.txt')
            with open(hex_filename, 'w') as f:
                hex_string = random_bytes.hex()
                # Formatear en líneas de 32 caracteres
                for i in range(0, len(hex_string), 32):
                    f.write(hex_string[i:i+32] + '\n')
            
            print(f"Archivo de bits aleatorios creado:")
            print(f"  Binario: {filename} ({num_bytes} bytes)")
            print(f"  Texto hex: {hex_filename}")
            
            return random_bytes
            
        except Exception as e:
            print(f"Error creando archivo de bits aleatorios: {e}")
            return None
    
    def bytes_to_bits(self, data_bytes):
        """Convierte bytes a array de bits"""
        bits = []
        for byte in data_bytes:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        return np.array(bits, dtype=int)
    
    def create_packet(self, data_bits):
        """Crea paquete con preámbulo, longitud, datos y checksum"""
        # Preámbulo para sincronización (patrón alternante + sincword)
        preamble = np.array([1,0,1,0,1,0,1,0] * 8 + [1,1,1,0,0,1,0,1])  # 72 bits
        
        # Longitud de datos (16 bits)
        length = len(data_bits)
        length_bits = [(length >> (15-i)) & 1 for i in range(16)]
        
        # Checksum simple (XOR de todos los bytes)
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                data_bytes.append(byte_val)
        
        checksum = 0
        for byte_val in data_bytes:
            checksum ^= byte_val
        
        checksum_bits = [(checksum >> (7-i)) & 1 for i in range(8)]
        
        # Paquete completo
        packet = np.concatenate([preamble, length_bits, data_bits, checksum_bits])
        return packet
    
    def transmit_file(self, filename, continuous=False, interval=2.0):
        """Transmite archivo por RF"""
        if not os.path.exists(filename):
            print(f"Generando archivo de bits aleatorios: {filename}")
            self.generate_random_bits_file(filename, 500)  # 500 bytes = 4000 bits
        
        # Leer archivo
        try:
            with open(filename, 'rb') as f:
                file_data = f.read()
            
            # Convertir a bits
            data_bits = self.bytes_to_bits(file_data)
            
            # Crear paquete
            packet_bits = self.create_packet(data_bits)
            
            # Modular
            modulated_signal = self.modem.modulate_qpsk(packet_bits)
            
            # Repetir señal para mejor recepción
            tx_signal = np.tile(modulated_signal, 3)  # Transmitir 3 veces por paquete
            
            print(f"Paquete preparado:")
            print(f"  Datos: {len(data_bits)} bits ({len(file_data)} bytes)")
            print(f"  Paquete total: {len(packet_bits)} bits")
            print(f"  Señal modulada: {len(tx_signal)} muestras")
            print(f"  Duración: {len(tx_signal)/self.sample_rate:.2f} segundos")
            
            # Transmisión
            if continuous:
                print(f"Transmisión continua cada {interval:.1f} segundos. Presiona Ctrl+C para parar.")
                try:
                    while True:
                        self.sdr.tx(tx_signal)
                        print(f"[{time.strftime('%H:%M:%S')}] Paquete transmitido")
                        time.sleep(interval)
                except KeyboardInterrupt:
                    print("\nTransmisión detenida por usuario")
            else:
                print("Transmitiendo paquete único...")
                self.sdr.tx(tx_signal)
                print("Transmisión completada")
                
        except Exception as e:
            print(f"Error en transmisión: {e}")

class PlutoSDRReceiver:
    """Clase para recepción RF con PlutoSDR en banda ISM"""
    
    def __init__(self, center_freq=433.92e6, sample_rate=250000):
        self.center_freq = int(center_freq)
        self.sample_rate = int(sample_rate)
        self.bandwidth = 25000
        self.modem = DigitalModem(self.bandwidth)
        
        # Historial para gráficas
        self.snr_history = deque(maxlen=200)
        self.ber_history = deque(maxlen=100)
        self.packet_count = 0
        self.error_count = 0
        
        try:
            # Conexión USB por defecto
            self.sdr = adi.Pluto("ip:192.168.2.1")
            
            # Configuración RX
            self.sdr.rx_rf_bandwidth = int(self.sample_rate)
            self.sdr.rx_lo = int(self.center_freq)
            self.sdr.gain_control_mode_chan0 = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 50  # dB (ajustable)
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_buffer_size = 2**15  # 32k muestras
            
            print(f"PlutoSDR RX configurado:")
            print(f"  Frecuencia: {self.center_freq/1e6:.3f} MHz (Banda ISM)")
            print(f"  Sample Rate: {self.sample_rate/1000:.0f} kHz")
            print(f"  Bandwidth: {self.bandwidth/1000:.0f} kHz")
            print(f"  Ganancia RX: 50 dB")
            
        except Exception as e:
            print(f"Error conectando PlutoSDR RX: {e}")
            print("Asegúrate de que PlutoSDR esté conectado por USB")
            sys.exit(1)
    
    def calculate_snr(self, signal):
        """Calcula SNR de la señal recibida"""
        if len(signal) == 0:
            return 0
        
        # Potencia de la señal
        signal_power = np.mean(np.abs(signal)**2)
        
        # Estimación del ruido usando percentil bajo
        power_samples = np.abs(signal)**2
        noise_floor = np.percentile(power_samples, 25)  # 25% percentile como ruido
        
        if noise_floor > 0 and signal_power > noise_floor:
            snr_linear = signal_power / noise_floor
            snr_db = 10 * np.log10(snr_linear)
            return max(0, min(40, snr_db))  # Limitar 0-40 dB
        
        return 0
    
    def find_preamble(self, bits):
        """Busca preámbulo en los bits recibidos"""
        if len(bits) < 72:
            return -1
        
        # Patrón del preámbulo
        preamble_pattern = np.array([1,0,1,0,1,0,1,0] * 8 + [1,1,1,0,0,1,0,1])
        
        best_correlation = 0
        best_position = -1
        
        # Buscar correlación
        for i in range(len(bits) - len(preamble_pattern)):
            if i + len(preamble_pattern) > len(bits):
                break
            
            correlation = np.sum(bits[i:i+len(preamble_pattern)] == preamble_pattern)
            correlation_ratio = correlation / len(preamble_pattern)
            
            if correlation_ratio > 0.75 and correlation > best_correlation:
                best_correlation = correlation
                best_position = i
        
        return best_position if best_correlation > 0.75 * len(preamble_pattern) else -1
    
    def decode_packet(self, bits, preamble_pos):
        """Decodifica paquete encontrado"""
        try:
            # Extraer longitud (16 bits después del preámbulo)
            length_start = preamble_pos + 72
            if length_start + 16 > len(bits):
                return None, "Paquete incompleto"
            
            length_bits = bits[length_start:length_start + 16]
            length = sum(bit << (15-i) for i, bit in enumerate(length_bits))
            
            # Extraer datos
            data_start = length_start + 16
            if data_start + length + 8 > len(bits):
                return None, "Datos incompletos"
            
            data_bits = bits[data_start:data_start + length]
            checksum_bits = bits[data_start + length:data_start + length + 8]
            
            # Verificar checksum
            received_checksum = sum(bit << (7-i) for i, bit in enumerate(checksum_bits))
            
            # Calcular checksum esperado
            expected_checksum = 0
            for i in range(0, len(data_bits), 8):
                if i + 8 <= len(data_bits):
                    byte_bits = data_bits[i:i+8]
                    byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                    expected_checksum ^= byte_val
            
            checksum_ok = (received_checksum == expected_checksum)
            
            return {
                'data_bits': data_bits,
                'length': length,
                'checksum_ok': checksum_ok,
                'received_checksum': received_checksum,
                'expected_checksum': expected_checksum
            }, None
            
        except Exception as e:
            return None, f"Error decodificando: {e}"
    
    def bits_to_bytes(self, bits):
        """Convierte bits a bytes"""
        bytes_data = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                bytes_data.append(byte_val)
        return bytes(bytes_data)
    
    def receive_and_analyze(self, output_filename, duration=60):
        """Recepción con análisis en tiempo real"""
        print(f"Iniciando recepción por {duration} segundos...")
        print("Se mostrarán gráficas en tiempo real. Cierra la ventana para terminar.")
        
        # Setup gráficas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # SNR plot
        snr_line, = ax1.plot([], [], 'b-', linewidth=2, label='SNR')
        ax1.set_xlim(0, 200)
        ax1.set_ylim(0, 40)
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('SNR (dB)')
        ax1.set_title('SNR en Tiempo Real')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Espectro/Waterfall
        ax2.set_xlabel('Frecuencia (kHz)')
        ax2.set_ylabel('Tiempo')
        ax2.set_title('Espectro Recibido')
        
        # Constelación
        ax3.set_xlim(-2, 2)
        ax3.set_ylim(-2, 2)
        ax3.set_xlabel('I (En Fase)')
        ax3.set_ylabel('Q (Cuadratura)')
        ax3.set_title('Constelación QPSK')
        ax3.grid(True, alpha=0.3)
        constellation_scatter = ax3.scatter([], [], alpha=0.6, s=10)
        
        # Estadísticas
        ax4.axis('off')
        ax4.set_title('Estadísticas de Recepción', pad=20)
        stats_text = ax4.text(0.05, 0.95, '', fontsize=11, transform=ax4.transAxes, 
                             verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Variables de estado
        received_packets = []
        stop_event = threading.Event()
        rx_queue = queue.Queue(maxsize=10)
        
        def rx_thread():
            """Hilo de recepción continua"""
            while not stop_event.is_set():
                try:
                    samples = self.sdr.rx()
                    if not rx_queue.full():
                        rx_queue.put(samples)
                except Exception as e:
                    print(f"Error RX: {e}")
                time.sleep(0.05)
        
        # Iniciar hilo RX
        rx_worker = threading.Thread(target=rx_thread, daemon=True)
        rx_worker.start()
        
        start_time = time.time()
        last_spectrum_update = 0
        spectrum_data = []
        
        def update_display(frame):
            nonlocal last_spectrum_update, spectrum_data
            
            current_time = time.time() - start_time
            
            # Obtener muestras
            if not rx_queue.empty():
                try:
                    samples = rx_queue.get_nowait()
                    
                    # Calcular SNR
                    snr = self.calculate_snr(samples)
                    self.snr_history.append(snr)
                    
                    # Actualizar SNR plot
                    if len(self.snr_history) > 1:
                        snr_line.set_data(range(len(self.snr_history)), list(self.snr_history))
                    
                    # Demodular
                    rx_bits = self.modem.demodulate_qpsk(samples)
                    
                    # Buscar paquetes
                    preamble_pos = self.find_preamble(rx_bits)
                    if preamble_pos >= 0:
                        packet_data, error = self.decode_packet(rx_bits, preamble_pos)
                        
                        if packet_data and packet_data['checksum_ok']:
                            self.packet_count += 1
                            received_packets.append(packet_data)
                            print(f"[{time.strftime('%H:%M:%S')}] Paquete #{self.packet_count} recibido correctamente")
                        elif packet_data:
                            self.error_count += 1
                            print(f"[{time.strftime('%H:%M:%S')}] Paquete con error de checksum")
                    
                    # Actualizar constelación
                    if len(samples) > 100:
                        symbol_samples = samples[::8][:200]  # Decimado para símbolos
                        constellation_scatter.set_offsets(np.column_stack([
                            symbol_samples.real, symbol_samples.imag
                        ]))
                    
                    # Actualizar espectro cada segundo
                    if current_time - last_spectrum_update > 1.0:
                        fft_data = np.fft.fftshift(np.fft.fft(samples))
                        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate)) / 1000  # kHz
                        spectrum_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-10)
                        
                        ax2.clear()
                        ax2.plot(freq_axis, spectrum_magnitude, 'g-', alpha=0.7)
                        ax2.set_xlabel('Frecuencia (kHz)')
                        ax2.set_ylabel('Magnitud (dB)')
                        ax2.set_title('Espectro Recibido')
                        ax2.grid(True, alpha=0.3)
                        
                        last_spectrum_update = current_time
                    
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Error en actualización: {e}")
            
            # Actualizar estadísticas
            packet_loss_rate = (self.error_count / max(1, self.packet_count + self.error_count)) * 100
            current_snr = self.snr_history[-1] if self.snr_history else 0
            avg_snr = np.mean(list(self.snr_history)) if self.snr_history else 0
            
            stats_text.set_text(f"""ESTADÍSTICAS DE RECEPCIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tiempo transcurrido:     {current_time:.1f} s
SNR actual:              {current_snr:.1f} dB
SNR promedio:            {avg_snr:.1f} dB

Paquetes correctos:      {self.packet_count}
Paquetes con error:      {self.error_count}
Tasa de pérdida:         {packet_loss_rate:.1f}%

Frecuencia central:      {self.center_freq/1e6:.3f} MHz
Ancho de banda:          {self.bandwidth/1000:.0f} kHz
Ganancia RX:             50 dB

Estado:                  {'Recibiendo...' if current_time < duration else 'Finalizado'}""")
            
            return snr_line, constellation_scatter, stats_text
        
        # Animación
        ani = animation.FuncAnimation(fig, update_display, interval=200, blit=False, cache_frame_data=False)
        
        # Mostrar gráficas
        plt.show(block=False)
        
        # Esperar duración o cierre de ventana
        for _ in range(int(duration * 10)):
            if not plt.get_fignums():  # Ventana cerrada
                break
            time.sleep(0.1)
        
        stop_event.set()
        plt.close('all')
        
        # Guardar datos recibidos
        if received_packets:
            self.save_received_data(received_packets, output_filename)
            print(f"\nRecepción completada. {len(received_packets)} paquetes guardados en {output_filename}")
        else:
            print("\nNo se recibieron paquetes válidos.")
    
    def save_received_data(self, packets, filename):
        """Guarda paquetes recibidos"""
        try:
            all_data = b''
            for packet in packets:
                packet_bytes = self.bits_to_bytes(packet['data_bits'])
                all_data += packet_bytes
            
            # Guardar archivo binario
            with open(filename, 'wb') as f:
                f.write(all_data)
            
            # Guardar versión hex
            hex_filename = filename.replace('.bin', '_received_hex.txt')
            with open(hex_filename, 'w') as f:
                hex_string = all_data.hex()
                for i in range(0, len(hex_string), 32):
                    f.write(hex_string[i:i+32] + '\n')
            
            print(f"Datos guardados: {filename} ({len(all_data)} bytes)")
            print(f"Versión hex: {hex_filename}")
            
        except Exception as e:
            print(f"Error guardando datos: {e}")

def compare_files(original_file, received_file):
    """Compara archivos original y recibido"""
    try:
        with open(original_file, 'rb') as f:
            original_data = f.read()
        
        with open(received_file, 'rb') as f:
            received_data = f.read()
        
        # Comparar byte a byte
        min_len = min(len(original_data), len(received_data))
        byte_errors = sum(1 for i in range(min_len) if original_data[i] != received_data[i])
        
        # Calcular BER
        bit_errors = 0
        for i in range(min_len):
            diff = original_data[i] ^ received_data[i]
            bit_errors += bin(diff).count('1')
        
        total_bits = min_len * 8
        ber = bit_errors / total_bits if total_bits > 0 else 1.0
        
        print(f"\n{'='*50}")
        print(f"COMPARACIÓN DE ARCHIVOS")
        print(f"{'='*50}")
        print(f"Archivo original:     {len(original_data)} bytes")
        print(f"Archivo recibido:     {len(received_data)} bytes")
        print(f"Bytes diferentes:     {byte_errors}")
        print(f"Bits erróneos:        {bit_errors}")
        print(f"BER:                  {ber:.2e}")
        print(f"Tasa de éxito:        {(1-ber)*100:.4f}%")
        print(f"{'='*50}")
        
        return ber
        
    except Exception as e:
        print(f"Error comparando archivos: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='PlutoSDR BER Test Application - Banda ISM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  Modo Transmisor (una vez):
    python pluto_ber_ism.py --mode tx --file datos_test.bin
  
  Modo Transmisor (continuo):
    python pluto_ber_ism.py --mode tx --continuous --interval 3
  
  Modo Receptor:
    python pluto_ber_ism.py --mode rx --duration 120 --file datos_test.bin

Configuración RF:
  - Banda ISM: 433.92 MHz
  - Ancho de banda: 25 kHz  
  - Conexión: RF entre PlutoSDRs
  - USB: 192.168.2.1 (por defecto)
        """)
    
    parser.add_argument('--mode', choices=['tx', 'rx'], 
                       help='Seleccionar modo: tx (transmisor) o rx (receptor)')
    parser.add_argument('--freq', type=float, default=433.92e6,
                       help='Frecuencia central en Hz (por defecto: 433.92 MHz - Banda ISM)')
    parser.add_argument('--sample-rate', type=float, default=250000,
                       help='Tasa de muestreo en Hz (por defecto: 250 kHz)')
    parser.add_argument('--file', default='datos_test.bin',
                       help='Archivo de datos (por defecto: datos_test.bin)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duración de recepción en segundos (por defecto: 60s)')
    parser.add_argument('--continuous', action='store_true',
                       help='Transmisión continua (solo modo TX)')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Intervalo entre transmisiones en segundos (por defecto: 2.0s)')
    parser.add_argument('--tx-power', type=int, default=-10,
                       help='Potencia de transmisión en dBm (por defecto: -10 dBm)')
    parser.add_argument('--rx-gain', type=int, default=50,
                       help='Ganancia de recepción en dB (por defecto: 50 dB)')
    
    args = parser.parse_args()
    
    # Si no se especifica modo, preguntar al usuario
    if not args.mode:
        print("\n" + "="*60)
        print("   PlutoSDR BER Test Application - Banda ISM")
        print("="*60)
        print("1. TX - Modo Transmisor")
        print("2. RX - Modo Receptor")
        print("="*60)
        
        while True:
            try:
                choice = input("Selecciona el modo (1 para TX, 2 para RX): ").strip()
                if choice == '1':
                    args.mode = 'tx'
                    break
                elif choice == '2':
                    args.mode = 'rx'
                    break
                else:
                    print("Por favor, ingresa 1 o 2.")
            except KeyboardInterrupt:
                print("\nOperación cancelada.")
                sys.exit(0)
    
    # Mostrar configuración
    print(f"\n{'='*60}")
    print(f"   CONFIGURACIÓN PlutoSDR BER TEST")
    print(f"{'='*60}")
    print(f"Modo:                 {args.mode.upper()}")
    print(f"Frecuencia:           {args.freq/1e6:.3f} MHz (Banda ISM)")
    print(f"Sample Rate:          {args.sample_rate/1000:.0f} kHz")
    print(f"Ancho de banda:       25 kHz")
    print(f"Conexión PlutoSDR:    USB (192.168.2.1)")
    print(f"Archivo:              {args.file}")
    
    if args.mode == 'tx':
        print(f"Potencia TX:          {args.tx_power} dBm")
        if args.continuous:
            print(f"Modo:                 Continuo (cada {args.interval}s)")
        else:
            print(f"Modo:                 Transmisión única")
    else:
        print(f"Ganancia RX:          {args.rx_gain} dB")
        print(f"Duración:             {args.duration} segundos")
    
    print(f"{'='*60}")
    
    if args.mode == 'tx':
        # Modo Transmisor
        print("\nInicializando transmisor...")
        try:
            tx = PlutoSDRTransmitter(args.freq, args.sample_rate)
            
            # Ajustar potencia si se especificó
            if args.tx_power != -10:
                tx.sdr.tx_hardwaregain_chan0 = args.tx_power
                print(f"Potencia TX ajustada a: {args.tx_power} dBm")
            
            print(f"\n🚀 INICIANDO TRANSMISIÓN")
            print(f"Presiona Ctrl+C para detener\n")
            
            tx.transmit_file(args.file, args.continuous, args.interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Transmisión detenida por el usuario")
        except Exception as e:
            print(f"❌ Error en transmisor: {e}")
    
    elif args.mode == 'rx':
        # Modo Receptor  
        print("\nInicializando receptor...")
        try:
            rx = PlutoSDRReceiver(args.freq, args.sample_rate)
            
            # Ajustar ganancia si se especificó
            if args.rx_gain != 50:
                rx.sdr.rx_hardwaregain_chan0 = args.rx_gain
                print(f"Ganancia RX ajustada a: {args.rx_gain} dB")
            
            print(f"\n📡 INICIANDO RECEPCIÓN")
            print(f"Duración: {args.duration} segundos")
            print(f"Se abrirán gráficas en tiempo real...\n")
            
            # Archivo de salida
            output_file = args.file.replace('.bin', '_received.bin')
            
            rx.receive_and_analyze(output_file, args.duration)
            
            # Comparar archivos si existe el original
            if os.path.exists(args.file):
                print(f"\nComparando con archivo original...")
                compare_files(args.file, output_file)
            else:
                print(f"\nArchivo original '{args.file}' no encontrado para comparación.")
                print(f"Para comparar, ejecuta primero el transmisor para generar el archivo.")
            
        except KeyboardInterrupt:
            print("\n🛑 Recepción detenida por el usuario")
        except Exception as e:
            print(f"❌ Error en receptor: {e}")
    
    print(f"\n{'='*60}")
    print("   Aplicación finalizada")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Aplicación terminada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
        return 1

if __name__ == "__main__":
    sys.exit(main())
