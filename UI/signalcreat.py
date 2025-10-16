import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d


def create_voice_function(file_path):
    """
    将语音文件转换为时间函数 voice_signal(t)
    :param file_path: 语音文件路径(.wav)
    :return: 函数对象 voice_signal(t)，t为时间(秒)
    """
    # 读取语音文件
    fs, data = wavfile.read(file_path)

    # 转换为单声道并归一化
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # 创建时间轴(秒)
    duration = len(data) / fs
    t_samples = np.linspace(0, duration, len(data))

    # 构建插值函数（支持向量化输入）
    voice_interp = interp1d(
        t_samples, data,
        kind='cubic',  # 三次样条插值保证平滑
        bounds_error=False,
        fill_value=0.0
    )

    # 包装为时间函数
    def voice_signal(t):
        """
        语音时间函数
        :param t: 时间(秒)，支持标量或numpy数组
        :return: 对应时刻的语音幅值
        """
        return voice_interp(t)

    return voice_signal

