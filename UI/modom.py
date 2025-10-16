import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from matplotlib import rcParams
import logging
import time
import pickle
from typing import List, Tuple

# 配置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SatModem')


class SatelliteModem:
    """卫星调制解调系统（集成Turbo码）"""

    def __init__(self, fs=10e6, fc=70e6, rf_freq=12e9, symbol_rate=1e6):
        """
        初始化调制解调器参数
        :param fs: 采样频率 (Hz)
        :param fc: 中频频率 (Hz)
        :param rf_freq: 射频频率 (Hz)
        :param symbol_rate: 符号率 (symbols/sec)
        """
        self.fs = fs
        self.fc = fc
        self.rf_freq = rf_freq
        self.symbol_rate = symbol_rate
        self.delay = 0.275  # 卫星传输时延(秒)
        self.samples_per_symbol = int(fs / symbol_rate)
        self._design_filters()

        # 检测节点数据
        self.detection_nodes = {
            'raw_bits': None,
            'encoded_bits': None,
            'modulated': None,
            'received': None,
            'demodulated': None
        }

    def _design_filters(self):
        """设计调制解调所需的滤波器"""
        # 基带成型滤波器 (根升余弦)
        self.bb_taps = signal.firwin(101, 0.5 * self.symbol_rate, fs=self.fs, window='hamming')

        # 中频带通滤波器
        self.if_bpf = signal.firwin(
            201, [self.fc - 1.5 * self.symbol_rate, self.fc + 1.5 * self.symbol_rate],
            pass_zero=False, fs=self.fs
        )

        # 均衡器 (简化)
        self.eq_taps = np.ones(11) / 11

    def _symbol_mapping(self, bits: List[int], mod_type: str) -> np.ndarray:
        """星座映射（支持QPSK/8PSK）"""
        if mod_type == 'qpsk':
            symbols = []
            for i in range(0, len(bits), 2):
                pair = bits[i:i + 2]
                if len(pair) < 2: pair += [0] * (2 - len(pair))
                angle = np.pi / 4 + int(''.join(map(str, pair)), 2) * np.pi / 2
                symbols.append(np.exp(1j * angle))
            return np.array(symbols)
        elif mod_type == '8psk':
            symbols = []
            for i in range(0, len(bits), 3):
                triplet = bits[i:i + 3]
                if len(triplet) < 3: triplet += [0] * (3 - len(triplet))
                angle = int(''.join(map(str, triplet)), 2) * np.pi / 4
                symbols.append(np.exp(1j * angle))
            return np.array(symbols)
        else:
            raise ValueError(f"Unsupported modulation: {mod_type}")

    def _symbol_demapping_to_llr(self, symbols: np.ndarray, mod_type: str) -> List[float]:
        """软解映射输出LLR（用于Turbo解码）"""
        if mod_type == 'qpsk':
            llrs = []
            for sym in symbols:
                # 计算每个比特的LLR
                llr0 = np.abs(sym - np.exp(1j * np.pi / 4)) ** 2 - np.abs(sym - np.exp(1j * 5 * np.pi / 4)) ** 2
                llr1 = np.abs(sym - np.exp(1j * np.pi / 4)) ** 2 - np.abs(sym - np.exp(1j * 3 * np.pi / 4)) ** 2
                llrs.extend([llr0, llr1])
            return llrs
        else:
            raise ValueError("Only QPSK supported for LLR demapping")

    def _pulse_shaping(self, symbols: np.ndarray) -> np.ndarray:
        """脉冲成型"""
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol, dtype=np.complex64)
        upsampled[::self.samples_per_symbol] = symbols
        return signal.lfilter(self.bb_taps, 1.0, upsampled)

    def _channel_impairments(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """添加信道损伤"""
        # AWGN噪声
        snr_linear = 10 ** (snr_db / 10)
        noise_power = np.mean(np.abs(signal) ** 2) / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        return signal + noise

    def modulate_with_turbo(self, turbo_bits: List[int], mod_type: str = 'qpsk', snr_db: float = 20) -> np.ndarray:
        """调制预生成的Turbo码"""
        # 记录检测节点
        self.detection_nodes['encoded_bits'] = turbo_bits

        # 星座映射
        symbols = self._symbol_mapping(turbo_bits, mod_type)

        # 脉冲成型
        shaped = self._pulse_shaping(symbols)
        self.detection_nodes['modulated'] = shaped

        # 上变频到中频
        t = np.arange(len(shaped)) / self.fs
        if_signal = shaped * np.exp(1j * 2 * np.pi * self.fc * t)

        # 中频滤波
        if_signal = signal.lfilter(self.if_bpf, 1.0, if_signal)

        # 上变频到射频并添加噪声
        rf_signal = if_signal * np.exp(1j * 2 * np.pi * self.rf_freq * t)
        noisy_rf = self._channel_impairments(rf_signal, snr_db)

        return noisy_rf

    def demodulate_with_turbo(self, rf_signal: np.ndarray, mod_type: str = 'qpsk') -> Tuple[List[int], List[int]]:
        """解调并Turbo解码"""
        # 下变频到中频
        t = np.arange(len(rf_signal)) / self.fs
        if_signal = rf_signal * np.exp(-1j * 2 * np.pi * self.rf_freq * t)

        # 中频滤波
        if_signal = signal.lfilter(self.if_bpf, 1.0, if_signal)

        # 下变频到基带
        bb_signal = if_signal * np.exp(-1j * 2 * np.pi * self.fc * t)

        # 匹配滤波
        filtered = signal.lfilter(self.bb_taps, 1.0, bb_signal)
        self.detection_nodes['received'] = filtered

        # 符号定时恢复（简化）
        symbols = filtered[::self.samples_per_symbol]

        # 均衡
        eq_symbols = signal.lfilter(self.eq_taps, 1.0, symbols)
        self.detection_nodes['demodulated'] = eq_symbols

        # 软解映射输出LLR
        llrs = self._symbol_demapping_to_llr(eq_symbols, mod_type)


        return llrs

    def plot_detection_nodes(self):
        """可视化检测节点"""
        plt.figure(figsize=(15, 10))

        # 编码比特直方图
        plt.subplot(2, 3, 1)
        plt.hist(self.detection_nodes['encoded_bits'], bins=[-0.5, 0.5, 1.5])
        plt.title('Turbo编码比特分布')

        # 调制信号时域
        plt.subplot(2, 3, 2)
        plt.plot(np.real(self.detection_nodes['modulated'][:200]))
        plt.title('调制信号时域 (I路)')

        # 接收信号星座图
        plt.subplot(2, 3, 3)
        received = self.detection_nodes['received'][::10]
        plt.scatter(np.real(received), np.imag(received), alpha=0.3)
        plt.title('接收信号星座图')

        # 解调后星座图
        plt.subplot(2, 3, 4)
        demod = self.detection_nodes['demodulated']
        plt.scatter(np.real(demod), np.imag(demod))
        plt.title('均衡后星座图')

        # 眼图
        plt.subplot(2, 3, 5)
        eye_matrix = self.detection_nodes['received'][:100 * self.samples_per_symbol]
        eye_matrix = eye_matrix.reshape(100, self.samples_per_symbol)
        plt.plot(eye_matrix.T.real, 'b-', alpha=0.1)
        plt.title('眼图 (I路)')

        plt.tight_layout()
        plt.show()


