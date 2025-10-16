import random

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import deque
import time


class SatelliteBeamController:
    def __init__(self, transmitters: List[str], ends: List[str]):
        """
        初始化波束控制器
        :param transmitters: 图中所有Transmitter节点ID（Transmitter1-7）
        :param ends: 图中所有终端节点ID（end1-5）
        """
        # 拓扑结构（根据图片标注建立连接关系）
        self.topology = {
            "Transmitter1": ["end1", "Transmitter2", "Transmitter7"],
            "Transmitter2": ["end2", "Transmitter1", "Transmitter3"],
            "Transmitter3": ["Transmitter2", "Transmitter4"],
            "Transmitter4": ["end4", "Transmitter3", "Transmitter5"],
            "Transmitter5": ["Transmitter4", "Transmitter6"],
            "Transmitter6": ["end5", "Transmitter5", "Transmitter7"],
            "Transmitter7": ["end3", "Transmitter6", "Transmitter1"]
        }

        # 硬件参数（相控阵天线）
        self.antenna = {
            "elements": 64,  # 天线阵元数量
            "max_beam": 8,  # 最大同时波束数
            "beamwidth": np.radians(5),  # 波束宽度(弧度)
            "freq": 12e9  # 载波频率(Hz)
        }

        # 资源池
        self.resource = {
            "time_slots": deque(range(10)),  # TDMA时隙(10ms/时隙)
            "freq_blocks": np.linspace(11.9e9, 12.1e9, 16),  # 16个频块
            "power_budget": 100.0  # 总功率预算(W)
        }

        # 动态状态记录
        self.beam_alloc = {}  # 当前波束分配表
        self.user_demand = {}  # 终端需求记录
        self.interference = {}  # 干扰矩阵

        # 初始化终端需求（模拟）
        for end in ends:
            self.user_demand[end] = {
                "throughput": np.random.uniform(1, 10),  # Mbps
                "priority": random.choice([1, 2, 3])  # QoS等级
            }

    def dynamic_beamforming(self, current_time: float):
        """
        动态波束成形核心算法（SDMA实现）
        """
        # 步骤1：计算最优波束指向（基于终端位置和需求）
        active_beams = {}
        for tx in self.topology:
            # 获取当前Transmitter连接的终端
            connected_ends = [n for n in self.topology[tx] if n.startswith('end')]

            if not connected_ends:
                continue

            # 波束选择算法（简化版：选择需求最高的终端）
            target_end = max(connected_ends, key=lambda x: self.user_demand[x]['priority'])

            # 生成波束参数（实际系统需计算方向矢量）
            active_beams[tx] = {
                "target": target_end,
                "direction": self._calc_beam_direction(tx, target_end),
                "freq": self._allocate_freq(tx),
                "power": self._allocate_power(target_end),
                "time_slot": current_time % 10  # TDMA时隙分配
            }

        # 步骤2：干扰协调管理
        self._manage_interference(active_beams)

        # 更新分配表
        self.beam_alloc = active_beams
        return active_beams

    def _calc_beam_direction(self, tx: str, end: str) -> np.ndarray:
        """
        计算波束指向矢量（简化版：随机生成方向）
        实际系统应结合卫星轨道参数和终端位置计算
        """
        return np.random.randn(3)  # 返回3D方向矢量

    def _allocate_freq(self, tx: str) -> float:
        """
        频率分配策略（避免相邻Transmitter同频）
        """
        neighbors = [n for n in self.topology[tx] if n.startswith('Transmitter')]
        used_freqs = [self.beam_alloc.get(n, {}).get('freq', 0) for n in neighbors]
        available = [f for f in self.resource['freq_blocks'] if f not in used_freqs]
        return random.choice(available) if available else random.choice(self.resource['freq_blocks'])

    def _allocate_power(self, end: str) -> float:
        """
        功率分配策略（基于终端需求）
        """
        demand = self.user_demand[end]['throughput']
        max_power = self.resource['power_budget'] / len(self.user_demand)
        return min(demand * 0.5, max_power)  # 线性分配

    def _manage_interference(self, beams: Dict):
        """
        干扰协调管理（计算SINR）
        """
        for tx1, beam1 in beams.items():
            sinr = []
            for tx2, beam2 in beams.items():
                if tx1 == tx2:
                    continue
                # 简化干扰计算：频率相同且波束夹角小于阈值则产生干扰
                if beam1['freq'] == beam2['freq'] and \
                        np.dot(beam1['direction'], beam2['direction']) > np.cos(self.antenna['beamwidth'] * 2):
                    interference = beam2['power'] / (1 + np.linalg.norm(beam1['direction'] - beam2['direction']))
                    sinr.append(interference)

            self.interference[tx1] = sum(sinr) if sinr else 0

    def visualize_topology(self):
        """绘制拓扑图（模拟原图风格）"""
        fig = plt.figure(figsize=(10, 6), facecolor='#0A1E3F')  # 深蓝背景
        ax = fig.add_subplot(111, projection='3d')

        # 绘制地球轮廓（简化版）
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='#1A3D7F', alpha=0.3)

        # 绘制Transmitter节点（白色圆形）
        tx_pos = {f'Transmitter{i}': np.random.randn(3) for i in range(1, 8)}
        for tx, pos in tx_pos.items():
            ax.scatter(*pos, c='white', s=100, edgecolors='black')
            ax.text(*pos, tx, color='white', fontsize=8)

        # 绘制终端节点（蓝色矩形）
        end_pos = {f'end{i}': np.random.randn(3) * 0.8 for i in range(1, 6)}
        for end, pos in end_pos.items():
            ax.scatter(*pos, c='blue', s=100, marker='s')
            ax.text(*pos, end, color='white', fontsize=8)

        # 绘制连接线（根据拓扑）
        for tx, links in self.topology.items():
            for link in links:
                if link.startswith('end'):
                    ax.plot(*zip(tx_pos[tx], end_pos[link]), 'w-', linewidth=0.5)
                else:
                    ax.plot(*zip(tx_pos[tx], tx_pos[link]), 'w--', linewidth=0.3)

        ax.set_axis_off()
        plt.title("Satellite Beamforming Topology", color='white')
        plt.tight_layout()
        plt.show()


# 示例运行
if __name__ == "__main__":
    # 初始化控制器（匹配图片中的节点）
    controller = SatelliteBeamController(
        transmitters=[f"Transmitter{i}" for i in range(1, 8)],
        ends=[f"end{i}" for i in range(1, 6)]
    )

    # 模拟动态波束控制
    for t in np.arange(0, 1, 0.1):  # 模拟1秒内10个时隙
        beams = controller.dynamic_beamforming(t)
        print(f"Time {t:.1f}s:")
        for tx, params in beams.items():
            print(f"  {tx} -> {params['target']} "
                  f"| Freq: {params['freq'] / 1e9:.2f}GHz "
                  f"| Power: {params['power']:.1f}W "
                  f"| Interf: {controller.interference.get(tx, 0):.2f}")

    # 可视化拓扑（模拟原图风格）
    controller.visualize_topology()