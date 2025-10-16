import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False


class PCM_DAC:
    """
    脉冲编码调制(PCM)数模转换模块
    实现数字信号到模拟信号的转换
    """

    def __init__(self, fs=8000, bits=8):
        """
        初始化DAC参数
        :param fs: 采样频率(Hz)，默认8kHz
        :param bits: 量化位数，默认8位
        """
        self.fs = fs  # 采样频率
        self.bits = bits  # 量化位数
        self.quant_levels = 2** bits  # 量化级数
        self.T = 1 / fs  # 采样周期(秒)

    def decode(self, binary_codes, dynamic_range=2.0):
        """
        二进制解码函数
        :param binary_codes: 二进制编码序列
        :param dynamic_range: 信号动态范围
        :return: 量化后的信号值
        """
        quant_step = dynamic_range / self.quant_levels
        min_level = - (2 ** (self.bits - 1)) * quant_step

        # 将二进制字符串转换为量化值
        quantized_values = []
        for code in binary_codes:
            # 将二进制字符串转换为整数
            index = int(code, 2)
            # 计算对应的量化值
            value = min_level + index * quant_step
            quantized_values.append(value)

        return np.array(quantized_values)

    def reconstruct(self, quantized_signal, cutoff_ratio=0.4):
        """
        信号重建函数（使用低通滤波恢复模拟信号）
        :param quantized_signal: 量化后的信号
        :param cutoff_ratio: 截止频率与Nyquist频率的比率
        :return: 重建的模拟信号
        """
        # 生成时间序列
        t = np.arange(len(quantized_signal)) * self.T

        # 使用线性插值初步重建
        from scipy.interpolate import interp1d
        interp_func = interp1d(t, quantized_signal, kind='linear', fill_value="extrapolate")

        # 更高密度的时间点用于重建
        highres_t = np.linspace(0, t[-1], 10 * len(t))
        interp_signal = interp_func(highres_t)

        # 设计低通滤波器去除高频成分
        nyquist = self.fs / 2
        cutoff = cutoff_ratio * nyquist

        # 使用巴特沃斯滤波器
        b, a = signal.butter(6, cutoff / nyquist, btype='low')
        filtered_signal = signal.filtfilt(b, a, interp_signal)

        return highres_t, filtered_signal

    def power_control(self, signal, target_power):
        """
        功率控制函数
        :param signal: 输入信号
        :param target_power: 目标功率(dB)
        :return: 功率调整后的信号
        """
        current_power = 10 * np.log10(np.mean(signal ** 2))
        gain = 10** ((target_power - current_power) / 20)
        return signal * gain

    def plot_spectrum(self, signal, title):
        """
        绘制信号频谱
        :param signal: 输入信号
        :param title: 图表标题
        """
        N = len(signal)
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, 1 / self.fs)[:N // 2]

        plt.figure(figsize=(10, 4))
        plt.plot(xf, 2 / N * np.abs(yf[0:N // 2]))
        plt.title(title)
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度")
        plt.grid()
        plt.show()

    def process(self, binary_codes, target_power=None, plot=True):
        """
        完整的DAC处理流程
        :param binary_codes: 输入的PCM二进制编码
        :param target_power: 目标功率(dB)
        :param plot: 是否绘制图表
        :return: 重建的模拟信号(时间序列和幅值)
        """
        # 1. 解码
        quantized_signal = self.decode(binary_codes)

        # 2. 重建模拟信号
        t, reconstructed = self.reconstruct(quantized_signal)

        # 3. 功率控制
        if target_power is not None:
            reconstructed = self.power_control(reconstructed, target_power)

        if plot:
            # 绘制原始量化信号和重建信号
            plt.figure(figsize=(12, 6))
            plt.stem(np.arange(len(quantized_signal)) * self.T,
                     quantized_signal, 'b', markerfmt='bo', basefmt=" ", label='量化信号')
            plt.plot(t, reconstructed, 'r-', linewidth=2, label='重建模拟信号')
            plt.title("数模转换结果对比")
            plt.xlabel("时间 (s)")
            plt.ylabel("幅度")
            plt.legend()
            plt.grid()
            plt.show()

            # 绘制频谱
            self.plot_spectrum(quantized_signal, "量化信号频谱")
            self.plot_spectrum(reconstructed[::10], "重建模拟信号频谱")  # 降采样显示

        return t, reconstructed

