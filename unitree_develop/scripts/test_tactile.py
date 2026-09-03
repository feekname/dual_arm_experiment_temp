#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-01
Description: G1手臂运动控制 + 触觉传感器实时显示与记录

功能：
  1. 输入目标关节角度，在指定时间内（默认10s）五次多项式平滑运动到目标位置
  2. 实时显示左右手触觉传感器数据（五指压力值、平均值、最大值）
  3. 记录全过程触觉数据到txt文件（含时间戳、左右五指压力）

依赖：
  - unitree_sdk2_python (G1控制)
  - pybullet (运动学计算)
  - rclpy (ROS 2 触觉订阅，可选；无ROS时仅做运动控制)
  - matplotlib (实时曲线，可选，--plot 参数启用)

用法示例：
  # 基本用法：指定左右臂目标关节角度，10s运动到位
  python test_tactile.py \
      --left_q  0.0 0.4 0.0 1.57 0.0 0.0 0.0 \
      --right_q 0.0 0.4 0.0 1.57 0.0 0.0 0.0

  # 自定义运动时间15s，启用实时曲线
  python test_tactile.py --left_q ... --right_q ... --duration 15 --plot

  # 指定网卡和输出文件名
  python test_tactile.py --left_q ... --right_q ... --interface enp89s0 --output my_test.txt

  # 仅运动控制，不订阅触觉（无ROS环境时）
  python test_tactile.py --left_q ... --right_q ... --no-ros
'''
import os
import sys
import time
import argparse
import threading
import numpy as np
import pybullet as p
from datetime import datetime

# G1 SDK 路径（experiment 目录的上一级是 g1_advanced_control）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server

# ================================================================
# ROS 2 触觉订阅（可选导入）
# ================================================================
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[警告] 未找到 rclpy (ROS 2)，触觉传感器订阅功能不可用")
    print("[提示] 可使用 --no-ros 参数跳过，或在 ROS 2 环境中运行")


# ================================================================
# 触觉数据容器（线程安全）
# ================================================================
class TactileData:
    """
    线程安全的触觉数据共享容器
    压力数据格式：5个手指 [thumb, index, middle, ring, pinky]
    """
    def __init__(self):
        self.left_pressure = np.zeros(5)
        self.right_pressure = np.zeros(5)
        self.left_timestamp = 0.0
        self.right_timestamp = 0.0
        self.left_received = False
        self.right_received = False
        self._lock = threading.Lock()
        # 消息字段名（用于调试显示）
        self.left_msg_type = ""
        self.right_msg_type = ""

    def update_left(self, pressure: np.ndarray, timestamp: float):
        with self._lock:
            self.left_pressure = pressure.copy()
            self.left_timestamp = timestamp
            self.left_received = True

    def update_right(self, pressure: np.ndarray, timestamp: float):
        with self._lock:
            self.right_pressure = pressure.copy()
            self.right_timestamp = timestamp
            self.right_received = True

    def get(self):
        """获取左右手指压力、时间戳、接收状态"""
        with self._lock:
            return (self.left_pressure.copy(), self.right_pressure.copy(),
                    self.left_timestamp, self.right_timestamp,
                    self.left_received, self.right_received)


# ================================================================
# ROS 2 触觉订阅节点
# ================================================================
if ROS2_AVAILABLE:
    class TactileSubscriber(Node):
        """
        自动检测左右手触觉 topic 并订阅
        自动探测消息类型，自适应提取五指压力数据
        """
        FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

        def __init__(self, tactile_data: TactileData):
            super().__init__('test_tactile_subscriber')
            self.tactile_data = tactile_data
            self._setup_subscriptions()

        def _setup_subscriptions(self):
            """自动检测 topic 名称和消息类型并创建订阅"""
            # 等待 ROS 2 网络中的 topic 出现
            time.sleep(1.5)
            topics = self.get_topic_names_and_types()

            left_topic = None
            right_topic = None
            left_type_str = None
            right_type_str = None

            for name, types in topics:
                if 'touch_status' in name:
                    if 'left' in name:
                        left_topic = name
                        left_type_str = types[0] if types else None
                    elif 'right' in name:
                        right_topic = name
                        right_type_str = types[0] if types else None

            # 兜底：用默认 topic 名
            if not left_topic:
                left_topic = '/left_hand/touch_status_126'
            if not right_topic:
                right_topic = '/right_hand/touch_status_127'

            print(f"\n[触觉] 左手 topic: {left_topic}")
            print(f"[触觉] 右手 topic: {right_topic}")

            from rosidl_runtime_py.utilities import get_message

            # 订阅左手
            if left_type_str:
                try:
                    LeftMsg = get_message(left_type_str)
                    self.create_subscription(
                        LeftMsg, left_topic,
                        lambda msg: self._touch_callback(msg, 'left'), 10)
                    self.tactile_data.left_msg_type = left_type_str
                    print(f"[触觉] 左手消息类型: {left_type_str}")
                    self._print_msg_structure(LeftMsg, 'left')
                except Exception as e:
                    print(f"[触觉] 左手订阅失败: {e}")
            else:
                print(f"[触觉] 警告: 无法获取左手 topic 类型，请确认节点已启动")

            # 订阅右手
            if right_type_str:
                try:
                    RightMsg = get_message(right_type_str)
                    self.create_subscription(
                        RightMsg, right_topic,
                        lambda msg: self._touch_callback(msg, 'right'), 10)
                    self.tactile_data.right_msg_type = right_type_str
                    print(f"[触觉] 右手消息类型: {right_type_str}")
                except Exception as e:
                    print(f"[触觉] 右手订阅失败: {e}")
            else:
                print(f"[触觉] 警告: 无法获取右手 topic 类型")

        def _print_msg_structure(self, msg_class, hand: str):
            """打印消息字段结构，帮助用户确认数据格式"""
            try:
                inst = msg_class()
                fields = []
                for f in dir(inst):
                    if not f.startswith('_') and not callable(getattr(inst, f)):
                        val = getattr(inst, f)
                        if hasattr(val, '__len__') and not isinstance(val, str):
                            fields.append(f"{f}=array[{len(val)}]")
                        else:
                            fields.append(f"{f}={type(val).__name__}")
                print(f"[触觉] {hand}手消息字段: {', '.join(fields)}")
            except Exception as e:
                print(f"[触觉] 打印消息结构失败: {e}")

        def _touch_callback(self, msg, hand: str):
            """触觉数据回调：提取五指压力并更新共享容器"""
            pressure = self._extract_finger_pressures(msg)
            now = time.time()
            if hand == 'left':
                self.tactile_data.update_left(pressure, now)
            else:
                self.tactile_data.update_right(pressure, now)

        def _extract_finger_pressures(self, msg) -> np.ndarray:
            """
            从触觉消息中自适应提取5个手指的压力值。
            按优先级尝试多种常见消息格式：
              1. Float32MultiArray.data (数组取前5个)
              2. pressure / normal_force / forces / force / data 数组字段
              3. thumb/index/middle/ring/pinky 命名字段
              4. finger0~finger4 编号字段
              5. 遍历所有数值字段取前5个
            """
            # 方法1: Float32MultiArray 风格的 .data 数组
            if hasattr(msg, 'data'):
                val = getattr(msg, 'data')
                if hasattr(val, '__len__') and len(val) >= 5:
                    try:
                        return np.array([float(v) for v in val[:5]])
                    except (ValueError, TypeError):
                        pass

            # 方法2: 常见数组字段名
            for field_name in ['pressure', 'normal_force', 'forces', 'force',
                               'touch_data', 'tactile', 'values']:
                if hasattr(msg, field_name):
                    val = getattr(msg, field_name)
                    if hasattr(val, '__len__') and len(val) >= 5:
                        try:
                            return np.array([float(v) for v in val[:5]])
                        except (ValueError, TypeError):
                            pass

            # 方法3: 拇指/食指/中指/无名指/小指 命名字段
            pressures = []
            for fname in self.FINGER_NAMES:
                found = False
                # 直接字段
                if hasattr(msg, fname):
                    val = getattr(msg, fname)
                    if isinstance(val, (int, float)):
                        pressures.append(float(val))
                        found = True
                    elif hasattr(val, '__len__') and len(val) > 0:
                        try:
                            pressures.append(float(val[0]))
                            found = True
                        except (ValueError, TypeError):
                            pass
                # 带前缀的字段 (如 left_thumb, finger_thumb)
                if not found:
                    for prefix in ['left_', 'right_', 'finger_', 'f_']:
                        fn = prefix + fname
                        if hasattr(msg, fn):
                            val = getattr(msg, fn)
                            if isinstance(val, (int, float)):
                                pressures.append(float(val))
                                found = True
                                break
                            elif hasattr(val, '__len__') and len(val) > 0:
                                try:
                                    pressures.append(float(val[0]))
                                    found = True
                                    break
                                except (ValueError, TypeError):
                                    pass
                if not found:
                    pressures.append(0.0)
            if any(p > 0 for p in pressures):
                return np.array(pressures)

            # 方法4: finger0 ~ finger4 编号字段
            pressures = []
            for i in range(5):
                fname = f'finger{i}'
                if hasattr(msg, fname):
                    val = getattr(msg, fname)
                    if isinstance(val, (int, float)):
                        pressures.append(float(val))
                    elif hasattr(val, '__len__') and len(val) > 0:
                        try:
                            pressures.append(float(val[0]))
                        except (ValueError, TypeError):
                            pressures.append(0.0)
                    else:
                        pressures.append(0.0)
                else:
                    pressures.append(0.0)
            if any(p > 0 for p in pressures):
                return np.array(pressures)

            # 方法5: 遍历所有数值型字段，取前5个标量
            numeric_vals = []
            for f in dir(msg):
                if f.startswith('_'):
                    continue
                val = getattr(msg, f)
                if callable(val):
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numeric_vals.append(float(val))
                elif hasattr(val, '__len__') and not isinstance(val, str):
                    try:
                        for v in val:
                            if isinstance(v, (int, float)):
                                numeric_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if len(numeric_vals) >= 5:
                return np.array(numeric_vals[:5])

            # 全部失败
            return np.zeros(5)


    def run_tactile_subscriber(tactile_data: TactileData):
        """在独立线程中运行 ROS 2 触觉订阅"""
        rclpy.init()
        node = TactileSubscriber(tactile_data)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


# ================================================================
# G1 状态数据类（基于 init_pos.py，保持兼容）
# ================================================================
class G1data:
    def __init__(self):
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        urdf_path = os.path.join(os.getcwd(), "description",
                                  "g1_7dof_brainco_hand.urdf")
        self.g1 = p.loadURDF(urdf_path, useFixedBase=True)
        self.control_list = []
        for i in range(p.getNumJoints(self.g1)):
            joint = p.getJointInfo(self.g1, i)
            if joint[2] == p.JOINT_REVOLUTE:
                self.control_list.append(joint[0])
        print("Control joints:", self.control_list)
        self.NbDofs = len(self.control_list)
        self.EEId = self.control_list[-1] + 1

        self.q = {'left_q': [0.0]*7, 'right_q': [0.0]*7}
        self.dq = {'left_dq': [0.0]*7, 'right_dq': [0.0]*7}
        self.ddq = {'left_ddq': [0.0]*7, 'right_ddq': [0.0]*7}
        self.torque = {'left_torque': [0.0]*7, 'right_torque': [0.0]*7}
        self.force = {'left_force': [0.0]*6, 'right_force': [0.0]*6}
        G1data.last_print = 0

    def status_callback(self, data):
        current_time = time.time()
        if current_time - self.last_print > 0.005:
            self.q['left_q'] = data['left_q']
            self.q['right_q'] = data['right_q']
            self.dq['left_dq'] = data['left_dq']
            self.dq['right_dq'] = data['right_dq']
            self.torque['left_torque'] = data['left_tau']
            self.torque['right_torque'] = data['right_tau']
            self.last_print = current_time


# ================================================================
# 实时曲线绘制器（可选，--plot 启用）
# ================================================================
class RealTimeTactilePlot:
    """matplotlib 实时曲线：左右手五指压力随时间变化"""
    def __init__(self, history_len: int = 300):
        import matplotlib
        matplotlib.use('TkAgg')  # 交互式后端
        import matplotlib.pyplot as plt
        self.plt = plt

        self.history_len = history_len
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

        self.fig, self.axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        self.fig.suptitle('Tactile Sensor Real-time Pressure', fontsize=13, fontweight='bold')

        self.left_lines = []
        self.right_lines = []
        for i in range(5):
            l1, = self.axes[0].plot([], [], color=self.colors[i], linewidth=1.2,
                                     label=self.finger_names[i])
            l2, = self.axes[1].plot([], [], color=self.colors[i], linewidth=1.2,
                                     label=self.finger_names[i])
            self.left_lines.append(l1)
            self.right_lines.append(l2)

        self.axes[0].set_ylabel('Pressure (raw)')
        self.axes[0].set_title('Left Hand', fontsize=11)
        self.axes[0].legend(loc='upper left', ncol=5, fontsize=8)
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].set_ylabel('Pressure (raw)')
        self.axes[1].set_xlabel('Time (samples)')
        self.axes[1].set_title('Right Hand', fontsize=11)
        self.axes[1].legend(loc='upper left', ncol=5, fontsize=8)
        self.axes[1].grid(True, alpha=0.3)

        self.left_history = np.zeros((history_len, 5))
        self.right_history = np.zeros((history_len, 5))
        self.x_data = np.arange(history_len)

        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, left_p: np.ndarray, right_p: np.ndarray):
        """更新曲线数据"""
        self.left_history = np.roll(self.left_history, -1, axis=0)
        self.left_history[-1] = left_p
        self.right_history = np.roll(self.right_history, -1, axis=0)
        self.right_history[-1] = right_p

        for i in range(5):
            self.left_lines[i].set_data(self.x_data, self.left_history[:, i])
            self.right_lines[i].set_data(self.x_data, self.right_history[:, i])

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        self.plt.close(self.fig)


# ================================================================
# 工具函数
# ================================================================
def quintic_interpolate(start: np.ndarray, goal: np.ndarray,
                        t: float, duration: float) -> np.ndarray:
    """五次多项式插值（Minimum Jerk），t∈[0, duration]"""
    ratio = np.clip(t / duration, 0.0, 1.0)
    s = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
    return start + (goal - start) * s


def print_tactile_status(left_p: np.ndarray, right_p: np.ndarray,
                          left_recv: bool, right_recv: bool,
                          exp_time: float):
    """
    在 shell 上实时打印触觉数据
    显示：五指压力、平均值、最大值、接触状态
    """
    finger_labels = ['Thumb', 'Index', 'Mid', 'Ring', 'Pinky']

    # 清行并打印
    print(f"\r  t={exp_time:6.2f}s | ", end="")

    # 左手
    if left_recv:
        l_avg = np.mean(left_p)
        l_max = np.max(left_p)
        l_max_idx = np.argmax(left_p)
        l_contact = "●" if l_max > 50 else "○"
        print(f"L{l_contact} [", end="")
        for i in range(5):
            print(f"{finger_labels[i][0]}:{left_p[i]:6.0f}", end=" ")
        print(f"] avg={l_avg:6.0f} max={l_max:6.0f}({finger_labels[l_max_idx]}) | ", end="")
    else:
        print("L: -- no data -- | ", end="")

    # 右手
    if right_recv:
        r_avg = np.mean(right_p)
        r_max = np.max(right_p)
        r_max_idx = np.argmax(right_p)
        r_contact = "●" if r_max > 50 else "○"
        print(f"R{r_contact} [", end="")
        for i in range(5):
            print(f"{finger_labels[i][0]}:{right_p[i]:6.0f}", end=" ")
        print(f"] avg={r_avg:6.0f} max={r_max:6.0f}({finger_labels[r_max_idx]})", end="")
    else:
        print("R: -- no data --", end="")

    print("   ", end="", flush=True)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='G1手臂运动控制 + 触觉传感器实时显示与记录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python test_tactile.py --left_q 0 0.4 0 1.57 0 0 0 --right_q 0 0.4 0 1.57 0 0 0
  python test_tactile.py --left_q ... --right_q ... --duration 15 --plot
  python test_tactile.py --left_q ... --right_q ... --no-ros --output test_run1.txt
        ''')
    parser.add_argument('--left_q', type=float, nargs=7, required=True,
                        metavar=('q1','q2','q3','q4','q5','q6','q7'),
                        help='左臂目标关节角度 (rad), 7个值')
    parser.add_argument('--right_q', type=float, nargs=7, required=True,
                        metavar=('q1','q2','q3','q4','q5','q6','q7'),
                        help='右臂目标关节角度 (rad), 7个值')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='运动到目标位置的时间 (s), 默认10s')
    parser.add_argument('--hold-time', type=float, default=20.0,
                        help='到达目标后保持并记录触觉数据的时间 (s), 默认20s')
    parser.add_argument('--interface', type=str, default='eth0',
                        help='G1通信网卡名称, 默认 eth0')
    parser.add_argument('--plot', action='store_true',
                        help='启用matplotlib实时曲线显示')
    parser.add_argument('--no-ros', action='store_true',
                        help='禁用ROS触觉订阅（仅做运动控制+数据记录占位）')
    parser.add_argument('--output', type=str, default=None,
                        help='数据保存文件名 (默认自动生成: tactile_YYYYMMDD_HHMMSS.txt)')
    parser.add_argument('--print-interval', type=float, default=0.1,
                        help='shell打印间隔 (s), 默认0.1s')
    args = parser.parse_args()

    # ============================================================
    # 初始化
    # ============================================================
    print("=" * 70)
    print("  G1 手臂运动 + 触觉传感器测试")
    print(f"  左臂目标:  {np.array(args.left_q)}")
    print(f"  右臂目标:  {np.array(args.right_q)}")
    print(f"  运动时间:  {args.duration}s")
    print(f"  保持记录:  {args.hold_time}s")
    print(f"  网卡:      {args.interface}")
    print(f"  实时曲线:  {'开启' if args.plot else '关闭'}")
    print(f"  ROS触觉:   {'禁用' if args.no_ros else '启用'}")
    print("=" * 70)

    # 输出文件名
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tactile_{timestamp}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    # 初始化 G1 数据（PyBullet）
    print("\n[初始化] 加载 PyBullet 模型...")
    g1_data = G1data()

    # 关节限位
    joint_lower_limits = []
    joint_upper_limits = []
    for j in g1_data.control_list:
        info = p.getJointInfo(g1_data.g1, j)
        joint_lower_limits.append(info[8])
        joint_upper_limits.append(info[9])

    # 初始化 G1 服务器
    print(f"[初始化] 连接 G1 (网卡: {args.interface})...")
    server = G1Server(network_interface=args.interface,
                      shm_name="dev/shm/6_axis_force_shm")
    server.subscribe_status(g1_data.status_callback)
    server.start()
    print("[初始化] G1 连接成功")

    # 获取初始位置
    time.sleep(0.5)
    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    # 目标位置
    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)

    # 目标位置限位检查
    for i in range(7):
        if i < len(joint_lower_limits):
            goal_l[i] = np.clip(goal_l[i], joint_lower_limits[i], joint_upper_limits[i])
            goal_r[i] = np.clip(goal_r[i], joint_lower_limits[i], joint_upper_limits[i])

    # ============================================================
    # 启动 ROS 2 触觉订阅（独立线程）
    # ============================================================
    tactile_data = TactileData()
    tactile_thread = None

    if ROS2_AVAILABLE and not args.no_ros:
        print("\n[触觉] 启动 ROS 2 触觉订阅线程...")
        tactile_thread = threading.Thread(
            target=run_tactile_subscriber, args=(tactile_data,), daemon=True)
        tactile_thread.start()
        # 等待订阅建立
        time.sleep(3.0)
    elif args.no_ros:
        print("\n[触觉] 已禁用 ROS 触觉订阅（--no-ros）")
    else:
        print("\n[触觉] ROS 2 不可用，触觉数据将记录为零")

    # ============================================================
    # 初始化实时曲线
    # ============================================================
    plotter = None
    if args.plot:
        try:
            plotter = RealTimeTactilePlot()
            print("[曲线] 实时曲线已启动")
        except Exception as e:
            print(f"[曲线] 启动失败: {e}")
            plotter = None

    # ============================================================
    # 打开数据记录文件
    # ============================================================
    # 文件格式：
    # 行首注释说明，然后每行:
    # time(s) phase left_thumb left_index left_middle left_ring left_pinky
    #          right_thumb right_index right_middle right_ring right_pinky
    #          left_avg left_max right_avg right_max
    data_file = open(output_path, 'w')
    data_file.write("# G1 手臂运动 + 触觉传感器数据记录\n")
    data_file.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    data_file.write(f"# 左臂目标关节角: {goal_l.tolist()}\n")
    data_file.write(f"# 右臂目标关节角: {goal_r.tolist()}\n")
    data_file.write(f"# 运动时间: {args.duration}s, 保持时间: {args.hold_time}s\n")
    data_file.write("# 列说明:\n")
    data_file.write("#   1: time (s)\n")
    data_file.write("#   2: phase (0=moving, 1=holding)\n")
    data_file.write("#   3-7:  left_thumb, left_index, left_middle, left_ring, left_pinky\n")
    data_file.write("#   8-12: right_thumb, right_index, right_middle, right_ring, right_pinky\n")
    data_file.write("#   13: left_avg, 14: left_max, 15: right_avg, 16: right_max\n")
    data_file.write("#" + "=" * 60 + "\n")

    # ============================================================
    # 主控制循环
    # ============================================================
    print("\n" + "=" * 70)
    print("  实验开始")
    print("=" * 70)

    t_step = 0.01  # 100Hz
    experiment_start = time.time()
    last_print_time = 0.0
    record_count = 0

    try:
        # ---------- 阶段1: 运动到目标位置 ----------
        print(f"\n[阶段1] 运动到目标位置 ({args.duration}s)...")
        move_start = time.time()

        while True:
            elapsed = time.time() - move_start
            exp_time = time.time() - experiment_start

            if elapsed >= args.duration:
                target_l = goal_l
                target_r = goal_r
                # server.manager.set_arm_poses(target_l.tolist(), target_r.tolist())
                break

            # 五次多项式插值
            target_l = quintic_interpolate(start_l, goal_l, elapsed, args.duration)
            target_r = quintic_interpolate(start_r, goal_r, elapsed, args.duration)

            # 发送指令（速度设为0，PD位置控制）
            # server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
            #                               [0.0]*7, [0.0]*7)
            print(f"\r  t={exp_time:.2f}s | Moving to target...", end="", flush=True)
            print(f"\r  t={exp_time:.2f}s | Moving to target: Left {target_l}, Right {target_r}", end="", flush=True)

            # 读取触觉数据
            lp, rp, lt, rt, lrecv, rrecv = tactile_data.get()

            # 记录数据
            l_avg = np.mean(lp) if lrecv else 0.0
            l_max = np.max(lp) if lrecv else 0.0
            r_avg = np.mean(rp) if rrecv else 0.0
            r_max = np.max(rp) if rrecv else 0.0
            data_file.write(f"{exp_time:.3f} 0 "
                           f"{lp[0]:.1f} {lp[1]:.1f} {lp[2]:.1f} {lp[3]:.1f} {lp[4]:.1f} "
                           f"{rp[0]:.1f} {rp[1]:.1f} {rp[2]:.1f} {rp[3]:.1f} {rp[4]:.1f} "
                           f"{l_avg:.1f} {l_max:.1f} {r_avg:.1f} {r_max:.1f}\n")
            record_count += 1

            # 打印
            if exp_time - last_print_time >= args.print_interval:
                print_tactile_status(lp, rp, lrecv, rrecv, exp_time)
                last_print_time = exp_time

            # 曲线更新
            if plotter:
                plotter.update(lp, rp)

            time.sleep(t_step)

        print(f"\n[阶段1] 到达目标位置，用时 {time.time()-move_start:.2f}s")

        # ---------- 阶段2: 保持位置，记录触觉数据 ----------
        print(f"\n[阶段2] 保持位置并记录触觉数据 ({args.hold_time}s)...")
        print("  提示: 可以在此阶段用手触摸指尖，观察触觉数据变化")
        hold_start = time.time()

        while True:
            elapsed = time.time() - hold_start
            exp_time = time.time() - experiment_start

            if elapsed >= args.hold_time:
                break

            # 保持目标位置
            # server.manager.set_arm_poses(goal_l.tolist(), goal_r.tolist(),
            #                               [0.0]*7, [0.0]*7)

            # 读取触觉数据
            lp, rp, lt, rt, lrecv, rrecv = tactile_data.get()

            # 记录数据
            l_avg = np.mean(lp) if lrecv else 0.0
            l_max = np.max(lp) if lrecv else 0.0
            r_avg = np.mean(rp) if rrecv else 0.0
            r_max = np.max(rp) if rrecv else 0.0
            data_file.write(f"{exp_time:.3f} 1 "
                           f"{lp[0]:.1f} {lp[1]:.1f} {lp[2]:.1f} {lp[3]:.1f} {lp[4]:.1f} "
                           f"{rp[0]:.1f} {rp[1]:.1f} {rp[2]:.1f} {rp[3]:.1f} {rp[4]:.1f} "
                           f"{l_avg:.1f} {l_max:.1f} {r_avg:.1f} {r_max:.1f}\n")
            record_count += 1

            # 打印
            if exp_time - last_print_time >= args.print_interval:
                print_tactile_status(lp, rp, lrecv, rrecv, exp_time)
                last_print_time = exp_time

            # 曲线更新
            if plotter:
                plotter.update(lp, rp)

            time.sleep(t_step)

        print(f"\n[阶段2] 保持记录完成")

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断，正在停止...")
    except Exception as e:
        print(f"\n\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ========================================================
        # 清理
        # ========================================================
        print("\n" + "=" * 70)
        print("  实验结束，正在清理...")
        print("=" * 70)

        # 关闭数据文件
        data_file.write(f"# 记录总帧数: {record_count}\n")
        data_file.close()
        print(f"[数据] 已保存 {record_count} 帧数据到: {output_path}")

        # 关闭曲线
        if plotter:
            plotter.close()
            print("[曲线] 实时曲线已关闭")

        # 停止 G1（保持当前位置）
        try:
            current = server.manager.get_current_arm_states()
            # server.manager.set_arm_poses(current["left_q"], current["right_q"])
            server.stop()
            print("[G1] 已停止，机器人保持在当前位置")
        except Exception as e:
            print(f"[G1] 停止异常: {e}")

        # 断开 PyBullet
        p.disconnect(g1_data.g1)
        print("[PyBullet] 已断开")

        print(f"\n[完成] 数据文件: {output_path}")
        print("[提示] 可用以下命令快速查看数据统计:")
        print(f"  python -c \"import numpy as np; d=np.loadtxt('{output_path}', comments='#'); print('帧数:', len(d)); print('左手平均压力均值:', np.mean(d[d[:,1]==1, 12])); print('右手平均压力均值:', np.mean(d[d[:,1]==1, 14]))\"")


if __name__ == "__main__":
    main()
