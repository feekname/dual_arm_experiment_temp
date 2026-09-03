'''
Author: pengfei 524560850@qq.com
Date: 2026-05-06 14:55:01
LastEditors: pengfei 524560850@qq.com
LastEditTime: 2026-05-07 20:26:42
FilePath: \code\experiment\single_arm_collision.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
import sys
import numpy as np
import pybullet as p
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server


class G1data:
    def __init__(self):
        # 在pybullet中初始化g1
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        urdf_path = os.path.join(os.getcwd(), "description", "g1_7dof_brainco_hand.urdf")
        self.g1 = p.loadURDF(urdf_path, useFixedBase=True)
        self.control_list = []
        for i in range(p.getNumJoints(self.g1)):
            joint = p.getJointInfo(self.g1, i)
            if joint[2] == p.JOINT_REVOLUTE:
                self.control_list.append(joint[0])
        print("Control joints:", self.control_list)
        self.NbDofs = len(self.control_list)
        self.EEId = self.control_list[-1] + 1

        # 初始化关节状态
        self.q = {'left_q': [0.0]*7, 'right_q': [0.0]*7}
        self.dq = {'left_dq': [0.0]*7, 'right_dq': [0.0]*7}
        self.ddq = {'left_ddq': [0.0]*7, 'right_ddq': [0.0]*7}
        self.torque = {'left_torque': [0.0]*7, 'right_torque': [0.0]*7}
        self.force = {'left_force': [0.0]*6, 'right_force': [0.0]*6}

        # 
        G1data.last_print = 0

    def status_callback(self, data):
        '''
        回调函数用于监控g1状态
        '''
        # if not hasattr(G1data, "last_print"):
        #     G1data.last_print = 0
        
        current_time = time.time()
        if current_time - self.last_print > 0.005:
            # print(f"Received status: {data}")
            self.q['left_q'] = data['left_q']
            self.q['right_q'] = data['right_q']   
            self.dq['left_dq'] = data['left_dq']
            self.dq['right_dq'] = data['right_dq']
            self.torque['left_torque'] = data['left_tau']
            self.torque['right_torque'] = data['right_tau'] 
            self.last_print = current_time

def main():
    # 创建g1数据实例
    g1_data = G1data()
    joint_lower_limits = []
    joint_upper_limits = []
    for j in g1_data.control_list:
        info = p.getJointInfo(g1_data.g1, j)
        joint_lower_limits.append(info[8])
        joint_upper_limits.append(info[9])

    print(f'joint_lower_limits: {joint_lower_limits}')
    print(f'joint_upper_limits: {joint_upper_limits}')

    # 创建服务器实例
    print("Starting G1Server...")
    interface = "eth0" if len(sys.argv) < 2 else sys.argv[1]
    server = G1Server(network_interface=interface, shm_name="dev/shm/6_axis_force_shm")
    
    # 声明变量
    t_step = 0.01 # 时间步长

    # --- 安全限幅阀值 ---
    MAX_QD = 5.0                    # 最大关节速度 (rad/s)
    MAX_DELTA_Q = 0.05              # 单步最大角度变化 (rad)

    # 监控g1 arm状态
    server.subscribe_status(g1_data.status_callback)
    server.start()
    print("Monitoring G1 arm state...")

    # 控制机械臂运动
    # 获取初始位置
    current = server.manager.get_current_arm_states()
    start_l = current["left_q"]
    start_r = current["right_q"]

    for i in range(g1_data.NbDofs):
        p.resetJointState(g1_data.g1, g1_data.control_list[i], g1_data.q['left_q'][i])

    pi = np.pi
    # init_l = [-0.8, 0.4, -0.7, 0.4, 0.0, -0.6, 0.4]
    # init_l = [-0.8, 0.2, -0.7, 0.8, 0.0, -0.2, 0.4]     # 创新点1
    init_l = [0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    init_r = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    init_tra_l = np.linspace(start_l, init_l, 100)
    init_tra_r = np.linspace(start_l, init_r, 100)
    it = 0
    while True:
        if it < 100:
            server.manager.set_arm_poses(init_tra_l[it], init_tra_r[it], np.zeros(7), np.zeros(7)) # 确保是list类型
            it += 1
        
        time.sleep(t_step)

    
if __name__ == "__main__":
    main()

