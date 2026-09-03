import rclpy
from rclpy.node import Node
# ========== 消息类型导入 ==========
# 如果运行报错，先执行 ros2 topic type /right_hand/touch_status_127 查看真实类型，替换下面的导入
from ros2_stark_msgs.msg import TouchStatus
# ==================================
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time
from datetime import datetime
import collections
import threading

# ========== 配置参数 ==========
LEFT_TOPIC = "/left_hand/touch_status_126"
RIGHT_TOPIC = "/right_hand/touch_status_127"
RECORD_FILE = f"dual_hand_touch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
WINDOW_SIZE = 100       # 曲线显示最近100个采样点
UPDATE_INTERVAL = 100   # 绘图刷新间隔，单位ms
FINGER_NAMES = ["thumb", "index finger", "middle finger", "ring finger", "little finger"]
COLORS = ['#ff4d4f', '#faad14', '#52c41a', '#1890ff', '#722ed1']
# ==============================

class DualHandTouchVisualizer(Node):
    def __init__(self):
        super().__init__("dual_hand_touch_vis")

        # 订阅左右手话题
        self.left_sub = self.create_subscription(
            TouchStatus, LEFT_TOPIC, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            TouchStatus, RIGHT_TOPIC, self.right_callback, 10
        )

        # 数据缓冲区：时间轴共用，左右手各存5根手指
        self.time_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.left_buffers = [collections.deque(maxlen=WINDOW_SIZE) for _ in range(5)]
        self.right_buffers = [collections.deque(maxlen=WINDOW_SIZE) for _ in range(5)]

        # CSV 初始化
        self.csv_file = open(RECORD_FILE, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ["时间戳(s)"] + \
                 [f"右手_{name}" for name in FINGER_NAMES] + \
                 [f"左手_{name}" for name in FINGER_NAMES]
        self.csv_writer.writerow(header)

        self.start_time = time.time()
        self._lock = threading.Lock()  # 保护数据读写

        # 创建画布：上下两个子图
        self.fig, (self.ax_right, self.ax_left) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True
        )
        self.right_lines = []
        self.left_lines = []

        # 右手子图
        for i, name in enumerate(FINGER_NAMES):
            line, = self.ax_right.plot([], [], label=name, color=COLORS[i], linewidth=1.5)
            self.right_lines.append(line)
        self.ax_right.set_title("右手 实时触觉法向力", fontsize=13)
        self.ax_right.set_ylabel("法向力 (原始ADC值)", fontsize=11)
        self.ax_right.legend(loc="upper left", ncol=5)
        self.ax_right.grid(True, alpha=0.3)

        # 左手子图
        for i, name in enumerate(FINGER_NAMES):
            line, = self.ax_left.plot([], [], label=name, color=COLORS[i], linewidth=1.5)
            self.left_lines.append(line)
        self.ax_left.set_title("左手 实时触觉法向力", fontsize=13)
        self.ax_left.set_xlabel("时间 (s)", fontsize=11)
        self.ax_left.set_ylabel("法向力 (原始ADC值)", fontsize=11)
        self.ax_left.legend(loc="upper left", ncol=5)
        self.ax_left.grid(True, alpha=0.3)

        plt.tight_layout()

    @staticmethod
    def _calc_avg_force(finger_data_list):
        """输入5个手指的数据列表，返回每个手指的平均法向力"""
        avg_forces = []
        for fd in finger_data_list:
            avg = (fd.normal_force1 + fd.normal_force2 + fd.normal_force3) / 3.0
            avg_forces.append(avg)
        return avg_forces

    def right_callback(self, msg):
        current_time = time.time() - self.start_time
        avg_forces = self._calc_avg_force(msg.data)
        with self._lock:
            self.time_buffer.append(current_time)
            for i in range(5):
                self.right_buffers[i].append(avg_forces[i])
            # 写入CSV：右手在前，左手补0（等左手回调更新）
            # 这里简化处理：左右手各自回调都写一行，以时间戳对齐；也可以改成等两边都到再写
            row = [round(current_time, 3)] + \
                  [round(f, 2) for f in avg_forces] + \
                  [""]*5  # 左手位留空，左手回调会补充
            self.csv_writer.writerow(row)

    def left_callback(self, msg):
        current_time = time.time() - self.start_time
        avg_forces = self._calc_avg_force(msg.data)
        with self._lock:
            for i in range(5):
                self.left_buffers[i].append(avg_forces[i])
            # 追加写入左手数据
            row = [round(current_time, 3)] + [""]*5 + [round(f, 2) for f in avg_forces]
            self.csv_writer.writerow(row)

    def update_plot(self, frame):
        with self._lock:
            if len(self.time_buffer) == 0:
                return self.right_lines + self.left_lines

            t_list = list(self.time_buffer)
            # 更新右手曲线
            for i in range(5):
                self.right_lines[i].set_data(t_list, list(self.right_buffers[i]))
            # 更新左手曲线
            for i in range(5):
                self.left_lines[i].set_data(t_list, list(self.left_buffers[i]))

            # 自适应Y轴
            all_right = [f for buf in self.right_buffers for f in buf]
            all_left = [f for buf in self.left_buffers for f in buf]
            all_vals = all_right + all_left
            if all_vals:
                y_min = min(all_vals) - 20
                y_max = max(all_vals) + 20
                self.ax_right.set_ylim(y_min, y_max)
                self.ax_left.set_ylim(y_min, y_max)

            self.ax_right.set_xlim(0, max(t_list) + 0.5)

        return self.right_lines + self.left_lines

    def run(self):
        ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=UPDATE_INTERVAL, blit=False
        )
        plt.show()  # 阻塞，直到关闭窗口

    def shutdown(self):
        self.csv_file.close()
        self.get_logger().info(f"数据已保存到: {RECORD_FILE}")
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualHandTouchVisualizer()

    try:
        # ROS 自旋放子线程，主线程跑 matplotlib
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

