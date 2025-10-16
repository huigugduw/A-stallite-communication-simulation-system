import numpy as np
from itertools import chain
import random


class TurboEncoder:
    """Turbo码编码器实现（基于RSC编码和交织）"""

    def __init__(self, interleaver_size=128):
        """
        :param interleaver_size: 交织深度（单位：比特）
        """
        self.interleaver_size = interleaver_size
        self.interleaver = self._build_interleaver()
        self.rsc1 = RSCEncoder()  # 第一个RSC编码器
        self.rsc2 = RSCEncoder()  # 第二个RSC编码器

    def _build_interleaver(self):
        """构建伪随机交织器（固定种子保证可重复性）"""
        indices = list(range(self.interleaver_size))
        random.seed(42)  # 固定随机种子
        random.shuffle(indices)
        return indices

    def _interleave(self, data):
        """执行交织操作"""
        return [data[i] for i in self.interleaver]

    def encode(self, input_bits):
        """
        Turbo编码（码率=1/3）
        修正后的版本：确保系统位、校验位1、校验位2交替排列
        """
        # 分块处理（每块interleaver_size比特）
        blocks = [input_bits[i:i + self.interleaver_size]
                  for i in range(0, len(input_bits), self.interleaver_size)]

        encoded_bits = []
        for block in blocks:
            # RSC编码器1（系统位 + 校验位1）
            sys_bits, parity1 = self.rsc1.encode(block)

            # 交织后通过RSC编码器2（校验位2）
            interleaved = self._interleave(block)
            _, parity2 = self.rsc2.encode(interleaved)

            # 交替排列系统位和校验位
            for s, p1, p2 in zip(sys_bits, parity1, parity2):
                encoded_bits.extend([s, p1, p2])

        return encoded_bits


class RSCEncoder:
    """修正后的RSC编码器（3位状态寄存器）"""

    def __init__(self):
        self.feedback_poly = 0o7  # 1 + D^2 + D^3
        self.forward_poly = 0o5  # 1 + D^3

    def encode(self, input_bits):
        sys_bits = []
        parity_bits = []
        state = 0  # 初始状态清零

        for b in input_bits:
            # 系统位直接输出
            sys_bits.append(b)

            # 计算反馈路径
            feedback = (state & self.feedback_poly) >> 2

            # 计算校验位
            parity = b ^ feedback
            parity_bits.append(parity)

            # 更新状态
            state = ((state << 1) | b) & 0b111  # 保留3位状态

        return sys_bits, parity_bits


def preprocess_data(binary_codes, interleaver_size=128):
    """
    预处理：将8位二进制字符串列表转换为Turbo编码器输入格式
    :param binary_codes: 80个8位二进制字符串（如 ["10101100", ...]）
    :param interleaver_size: 交织深度
    :return: 填充后的0/1列表（长度为interleaver_size的整数倍）
    """
    # 验证输入格式
    if not all(isinstance(code, str) and len(code) == 8 and set(code).issubset({'0', '1'}) for code in binary_codes):
        raise ValueError("所有元素必须是8位二进制字符串")

    # 合并所有二进制字符串并转换为0/1整数列表
    binary_stream = [int(bit) for code in binary_codes for bit in code]

    # 填充到interleaver_size的整数倍
    padding = (interleaver_size - (len(binary_stream) % interleaver_size)) % interleaver_size
    binary_stream += [0] * padding

    return binary_stream


def simulate_ad_output():
    """模拟AD模块输出：生成80个8位二进制编码"""
    return [format(np.random.randint(0, 256), '08b') for _ in range(80)]

