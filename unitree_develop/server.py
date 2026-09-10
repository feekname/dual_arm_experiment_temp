import threading
import time
import numpy as np
from typing import Callable, List
from config import ArmConfig, G1JointIndex
from manager import G1DualArmManager
from shm_handler import ForceSensorReader

class G1Server:
    def __init__(self, network_interface: str, shm_name: str = "/dev/shm/6_axis_force_shm"):
        self.config = ArmConfig()
        self.manager = G1DualArmManager(self.config)
        self.sensor = ForceSensorReader(shm_name)
        
        self.interface = network_interface
        self._subscribers: List[Callable] = []
        self._stop_event = threading.Event()
        self.emergency_locked = False
        
        # --- 安全限位定义 (弧度) ---
        # 这里的数值应根据 G1 硬件手册微调，示例给出一个保守范围
        self.joint_limits = {
            "shoulder_pitch": (-3.0, 3.0),
            "elbow": (-0.1, 2.5), 
            # 你可以根据需要添加更多关节的物理限位
        }
    
    def subscribe_status(self, callback: Callable[[dict], None]):
        self._subscribers.append(callback)

    def start(self):
        self.manager.start(self.interface)
        print('test1')
        self.sensor.connect()
        self._pub_thread = threading.Thread(target=self._status_publisher_loop, daemon=True)
        self._pub_thread.start()
        print("[Server] G1 服务端已就绪。安全限位已激活。")

    def stop(self):
        """
        受控停止：
        1. 停止状态发布
        2. 保持当前位置不动
        3. 停止 Manager 控制循环
        """
        if self._stop_event.is_set():
            return()
        
        print("[Server] 正在执行受控停止...")
        self._stop_event.set()
        
        # 1. 锁定当前状态
        current = self.manager.get_current_arm_states()
        self.manager.set_arm_poses(current["left_q"], current["right_q"])
        self.emergency_locked = True
        print("[Server] 机器人已锁定在当前位置。")

        # 2. 安全回收线程
        if hasattr(self, '_pub_thread') and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)

        # # 3. 停止Manager循环
        # self.manager.is_running = False
        self.sensor.close()
        print("[Server] 系统已安全关闭。")

    # --- 核心安全检查：软限位 ---
    def _security_check(self, target_q: List[float], arm_type: str) -> bool:
        """
        在指令发送前检查是否超出物理限位或存在速度突变
        """
        # 1. 检查角度范围 (此处仅为示例逻辑)
        for q in target_q:
            if np.isnan(q) or np.isinf(q):
                print(f"[Security] 错误：检测到非法数值 (NaN/Inf)！")
                return False

        # 2. 检查突变（例如两帧之间位移不能超过 0.3 弧度）
        current = self.manager.get_current_arm_states()
        current_q = current[f"{arm_type}_q"]
        
        max_step = 0.3 
        # for c, t in zip(current_q, target_q):
        #     if abs(c - t) > max_step:
        #         print(f"[Security] 警告：关节跳变过大 ({abs(c-t):.2f} rad)，拒绝执行！")
        #         return False
        
        return True

    def move_arm_to_sync(self, left_goal: List[float], right_goal: List[float], duration: float):
        """
        带安全检查的平滑移动
        """
        if self.emergency_locked:
            print("[Server] 紧急锁定中，无法执行移动指令！")
            return
        
        # 执行前先做一次终点检查
        if not self._security_check(left_goal, "left") or not self._security_check(right_goal, "right"):
            return

        start_time = time.time()
        current_state = self.manager.get_current_arm_states()
        start_l = current_state["left_q"]
        start_r = current_state["right_q"]

        while time.time() - start_time < duration and not self._stop_event.is_set():
            t = (time.time() - start_time) / duration
            interp_l = [s + (g - s) * t for s, g in zip(start_l, left_goal)]
            interp_r = [s + (g - s) * t for s, g in zip(start_r, right_goal)]
            
            # print('*************************')
            # print(interp_l)
            # print('*************************')
            # print(interp_r)
            # print('\n')
            # 实时更新目标
            self.manager.set_arm_poses(interp_l, interp_r)
            time.sleep(0.01)

        if not self._stop_event.is_set():
            self.manager.set_arm_poses(left_goal, right_goal)

    def move_arm_to_sync_quintic(self, left_goal: List[float], right_goal: List[float], duration: float):
        """
        使用五次多项式规划（Minimum Jerk）实现的平滑同步移动
        """
        if self.emergency_locked:
            print("[Server] 紧急锁定中，无法执行移动指令！")
            return
    
        # 1. 安全检查
        if not self._security_check(left_goal, "left") or not self._security_check(right_goal, "right"):
            return

        # 2. 记录起始状态
        current_state = self.manager.get_current_arm_states()
        start_l = np.array(current_state["left_q"])
        start_r = np.array(current_state["right_q"])
        goal_l = np.array(left_goal)
        goal_r = np.array(right_goal)
    
        # 3. 频率补偿循环准备
        control_dt = 0.01  # 期望 100Hz
        start_time = time.time()
        next_tick = start_time
    
        print(f"[Server] 开始五次多项式平滑移动，时长: {duration}s")

        while True:
            now = time.time()
            elapsed = now - start_time
        
            # 4. 计算归一化时间系数 (0.0 到 1.0)
            if elapsed >= duration:
                break
        
            # 防止停止事件触发
            if self._stop_event.is_set():
                break

            # 五次多项式插值核心公式
            # s(t) 从 0 增加到 1，速度和加速度在两头均为 0
            ratio = elapsed / duration
            s_t = 10 * (ratio**3) - 15 * (ratio**4) + 6 * (ratio**5)
        
            # 5. 计算当前时刻的期望构型
            interp_l = start_l + (goal_l - start_l) * s_t
            interp_r = start_r + (goal_r - start_r) * s_t
        
            # 6. 下发指令 (直接转成 list)
            self.manager.set_arm_poses(interp_l.tolist(), interp_r.tolist())
        
            # 7. 动态休眠：确保循环频率严格稳定在 100Hz
            next_tick += control_dt
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 8. 确保最终精准到达目标点
        if not self._stop_event.is_set():
            self.manager.set_arm_poses(left_goal, right_goal)
        print("[Server] 运动任务完成")


    def _status_publisher_loop(self):
        while not self._stop_event.is_set():
            state = self.manager.get_current_arm_states()
            state["force_sensor"] = self.sensor.read_force()
            for sub in self._subscribers:
                sub(state)
            time.sleep(0.01)
