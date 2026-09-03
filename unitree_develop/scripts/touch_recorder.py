import rclpy
from rclpy.node import Node
from ros2_stark_msgs.msg import TouchStatus
import csv
import time
from datetime import datetime
import threading
import sys
import select

# ========== 配置参数 ==========
LEFT_TOPIC = "/left_hand/touch_status_126"
RIGHT_TOPIC = "/right_hand/touch_status_127"
RECORD_FILE = f"dual_hand_touch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
PRINT_INTERVAL = 0.2  # 终端打印间隔，单位秒
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# ==============================

class DualHandTouchLogger(Node):
    def __init__(self):
        super().__init__("dual_hand_touch_logger")

        # 订阅左右手话题
        self.left_sub = self.create_subscription(
            TouchStatus, LEFT_TOPIC, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            TouchStatus, RIGHT_TOPIC, self.right_callback, 10
        )

        # 最新数据缓存
        self.latest_right = [0.0] * 5
        self.latest_left = [0.0] * 5
        self.frame_count = 0

        # CSV文件初始化
        self.csv_file = open(RECORD_FILE, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ["timestamp_s"] + \
                 [f"Right_{name}" for name in FINGER_NAMES] + \
                 [f"Left_{name}" for name in FINGER_NAMES]
        self.csv_writer.writerow(header)
        self.csv_file.flush()  # 立即写入表头

        self.start_time = time.time()
        self._lock = threading.Lock()
        self._running = True
        self._last_print_time = 0

    @staticmethod
    def _calc_avg_force(finger_data_list):
        """计算每根手指的平均法向力"""
        avg_forces = []
        for fd in finger_data_list:
            avg = fd.normal_force1
            avg_forces.append(avg)
        return avg_forces

    def right_callback(self, msg):
        if not self._running:
            return
        current_time = time.time() - self.start_time
        avg_forces = self._calc_avg_force(msg.data)
        
        with self._lock:
            self.latest_right = avg_forces
            self.frame_count += 1
            # 写入CSV并强制刷盘
            row = [round(current_time, 3)] + \
                  [round(f, 2) for f in self.latest_right] + \
                  [round(f, 2) for f in self.latest_left]
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # 控制打印频率
        if current_time - self._last_print_time >= PRINT_INTERVAL:
            self._print_status(current_time)
            self._last_print_time = current_time

    def left_callback(self, msg):
        if not self._running:
            return
        current_time = time.time() - self.start_time
        avg_forces = self._calc_avg_force(msg.data)
        
        with self._lock:
            self.latest_left = avg_forces
            self.frame_count += 1
            # 写入CSV并强制刷盘
            row = [round(current_time, 3)] + \
                  [round(f, 2) for f in self.latest_right] + \
                  [round(f, 2) for f in self.latest_left]
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # 控制打印频率
        if current_time - self._last_print_time >= PRINT_INTERVAL:
            self._print_status(current_time)
            self._last_print_time = current_time

    def _print_status(self, current_time):
        """终端同一行刷新状态，不刷屏"""
        with self._lock:
            right_str = "  ".join([f"{f:6.1f}" for f in self.latest_right])
            left_str = "  ".join([f"{f:6.1f}" for f in self.latest_left])
        
        sys.stdout.write("\r" + " " * 150 + "\r")
        sys.stdout.write(
            f"T:{current_time:6.2f}s | "
            f"R:[{right_str}] | "
            f"L:[{left_str}] | "
            f"Frames:{self.frame_count}"
        )
        sys.stdout.flush()

    def shutdown(self):
        self._running = False
        time.sleep(0.1)  # 等待在途回调执行完毕
        with self._lock:
            self.csv_file.close()
        print("\n" + "="*60)
        print(f"数据已保存到: {RECORD_FILE}")
        print(f"总接收数据帧数: {self.frame_count}")
        print("程序已安全退出")
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualHandTouchLogger()

    # ROS自旋放后台线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 启动提示
    print("="*60)
    print("Revo2Touch 双手触觉数据采集启动成功")
    print(f"右手话题: {RIGHT_TOPIC}")
    print(f"左手话题: {LEFT_TOPIC}")
    print(f"记录文件: {RECORD_FILE}")
    print("操作: 输入 q 回车退出 | Ctrl+C 强制退出")
    print("="*60)
    print("手指顺序: Thumb(拇指) Index(食指) Middle(中指) Ring(无名指) Pinky(小指)")
    print("-"*60)

    try:
        # 主线程非阻塞监听键盘输入
        while node._running:
            # 0.1秒超时检测输入，不阻塞数据接收
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'q':
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

