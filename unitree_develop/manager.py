# manager.py
import time
import numpy as np
from typing import List, Optional
from unitree_sdk2py.utils.thread import RecurrentThread

from config import ArmConfig, G1JointIndex
from driver import G1Driver

class G1DualArmManager:
    """
    双臂协作管理器
    负责：高频控制循环、双臂指令同步、力矩数据分发
    """
    def __init__(self, config: ArmConfig):
        self.config = config
        self.driver = G1Driver() # 获取单例驱动
        
        # 目标状态寄存器
        self.target_q = np.zeros(30)
        self.target_dq = np.zeros(30)
        self.target_kp = np.zeros(30)
        self.target_kd = np.zeros(30)
        self.target_tau_ff = np.zeros(30) # 前馈力矩，用于抵消重力或搬运补偿
        
        # 外部力矩/触觉数据接口（预留给你的传感器）
        self.external_force_data = {"left": None, "right": None}
        
        # 控制循环线程
        self._control_thread = RecurrentThread(
            interval=self.config.control_dt, 
            target=self._control_loop, 
            name="G1_Control_Loop"
        )
        
        self.is_running = False

    def start(self, network_interface: str):
        """启动控制系统"""
        print(f"[Manager] 正在初始化网卡: {network_interface}...")
        self.driver.init_comm(network_interface)
        
        print("[Manager] 等待机器人状态反馈...")
        while not self.driver.has_received_state:
            time.sleep(0.1)
        
        # 初始目标设为当前姿态，防止启动瞬间跳动
        with self.driver._lock:
            self.target_q = np.copy(self.driver.state.q)
            
        print("[Manager] 启动高频控制循环...")
        self.is_running = True
        self._control_thread.Start()

    def _control_loop(self):
        """
        核心实时控制循环 (运行在 RecurrentThread 中)
        """
        if not self.is_running:
            return

        # 1. 开启 arm_sdk 控制权 (权重位设为1)
        self.driver.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

        # 2. 遍历所有受控关节（左右臂 + 腰部）
        controlled_joints = self.config.left_arm_idx + self.config.right_arm_idx + self.config.waist_idx

        for idx in controlled_joints:
            # 填入指令数据
            self.driver.low_cmd.motor_cmd[idx].mode = 1
            self.driver.low_cmd.motor_cmd[idx].q = self.target_q[idx]
            self.driver.low_cmd.motor_cmd[idx].dq = self.target_dq[idx] # 搬运任务通常设为0，由内环控制速度
            self.driver.low_cmd.motor_cmd[idx].kp = self.config.kp
            self.driver.low_cmd.motor_cmd[idx].kd = self.config.kd
            # 本项目当前使用纯位置控制：显式清零，避免命令缓冲区残留
            # 任何历史前馈力矩。静差由上层位置命令偏置消除。
            self.driver.low_cmd.motor_cmd[idx].tau = 0.0

        # 3. 发布指令
        self.driver.publish_cmd()

    # --- 高层 API 接口 ---

    def set_arm_poses(self, left_q: List[float], right_q: List[float], left_dq: List[float], right_dq: List[float]):
        """
        同时设置左右手的目标角度（同步更新）
        """
        if len(left_q) != len(self.config.left_arm_idx) or len(right_q) != len(self.config.right_arm_idx):
            print("[Error] 角度数组长度不匹配")
            return

        # 更新目标寄存器
        for i, idx in enumerate(self.config.left_arm_idx):
            self.target_q[idx] = left_q[i]
            self.target_dq[idx] = left_dq[i]
        for i, idx in enumerate(self.config.right_arm_idx):
            self.target_q[idx] = right_q[i]
            self.target_dq[idx] = right_dq[i]
        
        # print('***************************')
        # print(f'发布指令： 左手： {left_q}')

    def get_current_arm_states(self):
        """获取当前双臂的实时反馈（用于闭环控制或搬运状态判定）"""
        with self.driver._lock:
            left_states = [self.driver.state.q[i] for i in self.config.left_arm_idx]
            right_states = [self.driver.state.q[i] for i in self.config.right_arm_idx]
            left_states_dq = [self.driver.state.dq[i] for i in self.config.left_arm_idx]
            right_states_dq = [self.driver.state.dq[i] for i in self.config.right_arm_idx]
            left_states_ddq = [self.driver.state.ddq[i] for i in self.config.left_arm_idx]
            right_states_ddq = [self.driver.state.ddq[i] for i in self.config.right_arm_idx]
            left_tau = [self.driver.state.tau[i] for i in self.config.left_arm_idx]
            right_tau = [self.driver.state.tau[i] for i in self.config.right_arm_idx]
            
        return {
            "left_q": left_states, "right_q": right_states,
            "left_dq": left_states_dq, "right_dq": right_states_dq,
            "left_ddq": left_states_ddq, "right_ddq": right_states_ddq,
            "left_tau": left_tau, "right_tau": right_tau
        }

    def update_external_sensor(self, side: str, data: any):
        """
        供外部脚本调用，将你的触觉或力矩传感器数据同步进来
        """
        if side in ["left", "right"]:
            self.external_force_data[side] = data
