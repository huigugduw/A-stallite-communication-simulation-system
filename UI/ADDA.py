import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 设置支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PCM_ADC:
    """
    脉冲编码调制(PCM)模数转换模块
    实现模拟语音信号到数字信号的转换
    """

    def __init__(self, fs=8000, bits=8):
        """
        初始化ADC参数
        :param fs: 采样频率(Hz)，默认8kHz
        :param bits: 量化位数，默认8位
        """
        self.fs = fs  # 采样频率
        self.bits = bits  # 量化位数
        self.quant_levels = 2** bits  # 量化级数
        self.T = 1 / fs  # 采样周期(秒)

    def sample(self, analog_signal, t):
        """
        采样函数 - 按照Nyquist定理进行采样
        :param analog_signal: 模拟信号函数
        :param t: 时间序列
        :return: 采样后的离散时间信号
        """
        # 生成采样时间点
        sample_times = np.arange(0, t[-1], self.T)
        # 在采样点获取信号值
        sampled_signal = analog_signal(sample_times)
        return sample_times, sampled_signal

    def quantize(self, sampled_signal):
        """
        均匀量化函数
        :param sampled_signal: 采样后的信号
        :return: 量化后的信号和量化误差
        """
        # 确定信号动态范围
        max_amp = np.max(np.abs(sampled_signal))
        dynamic_range = 2 * max_amp

        # 计算量化间隔(步长)
        quant_step = dynamic_range / self.quant_levels

        # 执行量化
        quantized = np.round(sampled_signal / quant_step) * quant_step

        # 计算量化误差
        quantization_error = sampled_signal - quantized

        return quantized, quantization_error, quant_step

    def encode(self, quantized_signal, quant_step):
        min_level = - (2** (self.bits - 1)) *quant_step
        max_level = -min_level - quant_step  # 修正最大值

        # 限制索引在[0, 2^bits-1]范围内
        code_indices = np.clip(
            ((quantized_signal - min_level) / quant_step).astype(int),
            0, self.quant_levels - 1
        )
        return [format(idx, f'0{self.bits}b') for idx in code_indices]

    def analyze_quantization_error(self, quantization_error):
        """
        量化误差分析
        :param quantization_error: 量化误差序列
        """
        # 计算统计特性
        mean_error = np.mean(quantization_error)
        rms_error = np.sqrt(np.mean(quantization_error ** 2))
        max_error = np.max(np.abs(quantization_error))

        print("\n量化误差分析:")
        print(f"平均量化误差: {mean_error:.6f}")
        print(f"RMS量化误差: {rms_error:.6f}")
        print(f"最大量化误差: {max_error:.6f}")


        # 绘制误差分布
        """plt.figure(figsize=(10, 4))
        plt.hist(quantization_error, bins=50)
        plt.title("量化误差分布")
        plt.xlabel("误差值")
        plt.ylabel("频次")
        plt.grid(True)
        plt.show()"""

    def plot_fft(self, signal, title):
        """
        绘制信号的频谱
        :param signal: 输入信号
        :param title: 图表标题
        """
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / self.fs)[:N // 2]

        """plt.figure(figsize=(10, 4))
        plt.plot(xf, 2 / N * np.abs(yf[0:N // 2]))
        plt.title(title)
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度")
        plt.grid()
        plt.show()"""

    def process(self, analog_signal, duration=0.05, plot=True):
        """
        完整的PCM处理流程
        :param analog_signal: 模拟信号函数
        :param duration: 信号持续时间(秒)
        :param plot: 是否绘制图表
        :return: 二进制编码序列
        """
        # 生成时间序列
        t = np.linspace(0, duration, 10000)

        # 1. 采样
        sample_times, sampled_signal = self.sample(analog_signal, t)

        # 2. 量化
        quantized, quantization_error, quant_step = self.quantize(sampled_signal)

        # 3. 编码
        binary_codes = self.encode(quantized, quant_step)

        # 分析量化误差
        self.analyze_quantization_error(quantization_error)

        if plot:
            # 绘制原始信号和采样信号
            plt.figure(figsize=(12, 6))
            plt.plot(t, analog_signal(t), 'b-', label='原始模拟信号')
            plt.stem(sample_times, sampled_signal, 'r', markerfmt='ro', basefmt=" ", label='采样点')
            plt.title("采样过程 (fs=8kHz)")
            plt.xlabel("时间 (s)")
            plt.ylabel("幅度")
            plt.legend()
            plt.grid()
            plt.show()

            # 绘制量化过程
            plt.figure(figsize=(12, 6))
            plt.stem(sample_times, sampled_signal, 'b', markerfmt='bo', basefmt=" ", label='采样值')
            plt.step(sample_times, quantized, 'r', where='post', label='量化值')
            plt.title("量化过程 (8-bit)")
            plt.xlabel("时间 (s)")
            plt.ylabel("幅度")
            plt.legend()
            plt.grid()
            plt.show()

            # 绘制频谱
            """self.plot_fft(sampled_signal, "采样信号频谱")
            self.plot_fft(quantized, "量化后信号频谱")
            self.plot_fft(quantization_error, "量化误差频谱")"""

        return binary_codes


