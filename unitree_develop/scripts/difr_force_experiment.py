'''
Author: pengfei 524560850@qq.com
Date: 2026-09-08
Description: 双臂协作搬运实验 — DIFR (力传感器版本)
             用六维力传感器(共享内存)替换触觉传感器，实现公式41的DIFR验证

与触觉版本的区别:
  - 内力估计: 用力传感器Fz分量 (左右夹持力的平均值)
  - 外力估计: 保持GMO观测器不变
  - 去掉ROS 2触觉子进程，直接读共享内存
  - 去掉灵巧手控制（假设已手动夹持或用其他方式固定物体）

共享内存数据布局 (12个double):
  data[0:6]  = 左手六维力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
  data[6:12] = 右手六维力/力矩 [Fx, Fy, Fz, Mx, My, Mz]

实验三阶段:
  阶段1 MOVE  (0~10s): 双臂五次多项式运动到搬运预备姿态
  阶段2 GRASP (10~20s): 保持姿态，GMO稳态偏置采集，内力跟踪F_0
  阶段3 TEST  (20~50s): 碰撞测试，DIFR激活动态调节期望内力

DIFR核心算法（公式41 QP解析解）:
  滑动动力学: δ̇ = δ̇_prev + (F_E - μ·F_I)/m_o · Δt
              δ = δ_prev + δ̇ · Δt
  QP问题: min α0·(F_I-F_0)² + α1·δ²
  解析解: F_I* = (α0·F_0 + α1·k·C) / (α0 + α1·k²)
          k = μ·Δt²/m_o, C = δ_prev + δ̇_prev·Δt + F_E/m_o·Δt²

用法:
  python difr_force_experiment.py \
      --left_q  -0.8 0.7 -0.7 0.4 0.0 -0.6 0.0 \
      --right_q  0.8 0.7  0.7 0.4 0.0  0.6 0.0 \
      --shm-name 6_axis_force_shm \
      --F0 16.35 --mu 0.3 --m-object 1.0
'''
import os
import sys
import time
import argparse
import numpy as np
import pybullet as p
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server
from shm_handler import ForceSensorReader


# ================================================================
# 力传感器数据容器
# ================================================================
class ForceData:
    '''
    六维力传感器数据容器
    左右各6维: [Fx, Fy, Fz, Mx, My, Mz]
    '''
    AXIS = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, normal_axis='z', tangent_axis='y',
                 normal_sign=1.0, tangent_sign=1.0):
        self.left = np.zeros(6)   # 左手六维力
        self.left_bias = np.zeros(6)   # 左手零偏
        self.bias_applied = False
        self.normal_idx = self.AXIS[normal_axis]
        self.tangent_idx = self.AXIS[tangent_axis]
        self.normal_sign = normal_sign
        self.tangent_sign = tangent_sign

    def update(self, raw_data):
        '''
        从共享内存读取的12维原始数据更新
        Args:
            raw_data: 12维数组 [left_6dof, right_6dof]
        '''
        raw_data = np.asarray(raw_data, dtype=float).reshape(-1)
        if raw_data.size < 6:
            raise ValueError(f"左手六维力数据长度不足: {raw_data.size}")
        self.left = raw_data[:6].copy()

    def get_forces(self):
        '''获取去偏后的六维力'''
        l = self.left - self.left_bias if self.bias_applied else self.left
        return l

    def get_internal_force(self):
        '''
        估计内力（夹持力）
        夹持物体时，左右指尖的法向力(Fz)方向相反
        内力 = (|Fz_left| + |Fz_right|) / 2
        简化：只考虑单臂一边，用左手Fz的绝对值
        '''
        l = self.get_forces()
        signed_normal = self.normal_sign * l[self.normal_idx]
        return max(0.0, signed_normal), signed_normal

    def get_external_force_from_sensor(self):
        '''
        从力传感器估计外力（切向力，导致滑动的力）
        用左右Fx,Fy的合力作为外力估计
        '''
        l = self.get_forces()
        return self.tangent_sign * l[self.tangent_idx]


# ================================================================
# GMO 观测器
# ================================================================
class GMOObserver:
    def __init__(self, n_dof=7, gain=5.0):
        self.n_dof = n_dof
        self.K = gain * np.eye(n_dof)
        self.P_hat = np.zeros(n_dof)
        self.r = np.zeros(n_dof)
        self.M_last = None
        self._initialized = False

    def reset(self):
        self.P_hat = np.zeros(self.n_dof)
        self.r = np.zeros(self.n_dof)
        self.M_last = None
        self._initialized = False

    def update(self, M, Cq, G, tau, dq, dt):
        P = M @ dq
        if self._initialized and self.M_last is not None:
            dM = (M - self.M_last) / max(dt, 1e-6)
        else:
            dM = np.zeros_like(M)
        CTq = dM @ dq - Cq
        self.M_last = M.copy()
        self._initialized = True
        P_hat_dot = CTq + self.r + tau - G
        self.P_hat += P_hat_dot * dt
        self.r = self.K @ (P - self.P_hat)
        return self.r.copy()


# ================================================================
# DIFR 控制器（论文公式41 QP解析解）
# ================================================================
class DIFRController:
    '''
    Dynamic Internal Force Regulation — 基于滑动动力学的QP问题(公式41)
    决策变量: F_I (期望内力，标量，法向夹持力)
    目标: min α0·(F_I - F_0)² + α1·δ²
    约束: 滑动动力学(公式39-40), δ限位, F_I限位
    '''
    def __init__(self, F_0=16.35, F_min=5.0, F_max=40.0,
                 mu=0.3, m_object=1.0, alpha0=1.0, alpha1=1e8,
                 delta_min=-0.05, delta_max=0.05,
                 collision_threshold=8.0, fc_threshold=0.8, K_f=0.002):
        self.F_0 = F_0
        self.F_min = F_min
        self.F_max = F_max
        self.mu = mu
        self.m_object = m_object
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.collision_threshold = collision_threshold
        self.fc_threshold = fc_threshold
        self.K_f = K_f

        self.delta = 0.0
        self.delta_dot = 0.0
        self.F_I_des = F_0
        self.active = False

    def reset(self):
        self.delta = 0.0
        self.delta_dot = 0.0
        self.F_I_des = self.F_0
        self.active = False

    def update(self, F_E, F_I_est, gmo_force_norm, dt):
        '''
        DIFR更新 — 解公式41的QP问题
        Returns:
            F_I_des, delta_roll, delta, fc, active
        '''
        dt = float(np.clip(dt, 1e-4, 0.05))
        force_demand = abs(F_E)
        fc = force_demand / max(self.mu * self.F_I_des, 1e-6)

        collision = gmo_force_norm > self.collision_threshold
        slip = abs(self.delta) > 0.005 or abs(self.delta_dot) > 0.02
        friction_margin_low = fc > self.fc_threshold
        self.active = collision or slip or friction_margin_low

        if self.active:
            # 标量化公式(41)：先求无约束最优解，再投影到全部可行区间。
            k = self.mu * dt**2 / self.m_object
            C = self.delta + self.delta_dot * dt + force_demand / self.m_object * dt**2
            denom = self.alpha0 + self.alpha1 * k**2
            F_I_star = (self.alpha0 * self.F_0 + self.alpha1 * k * C) / denom

            # δ_min <= C-kF <= δ_max，以及保守摩擦锥约束 fc<=fc_threshold。
            slip_lower = (C - self.delta_max) / k
            friction_lower = force_demand / max(self.mu * self.fc_threshold, 1e-6)
            feasible_lower = max(self.F_min, slip_lower, friction_lower)
            feasible_upper = self.F_max
            if feasible_lower <= feasible_upper:
                F_I_des = np.clip(F_I_star, feasible_lower, feasible_upper)
            else:
                # 冲击过大、F_max不足时优先执行安全上限，并由日志中的fc>1暴露不可行。
                F_I_des = self.F_max

            # 更新滑动状态
            net_force = force_demand - self.mu * F_I_des
            if self.delta_dot <= 0.0 and net_force <= 0.0:
                self.delta_dot = 0.0  # 静摩擦区
            else:
                self.delta_dot = max(
                    0.0, self.delta_dot + net_force / self.m_object * dt)
            self.delta = min(self.delta + self.delta_dot * dt, self.delta_max)
        else:
            F_I_des = self.F_0
            self.delta *= 0.95
            self.delta_dot *= 0.95

        self.F_I_des = F_I_des
        fc = force_demand / max(self.mu * F_I_des, 1e-6)

        # 位置控制下的内力跟踪：肩关节roll相对偏移
        delta_roll = self.K_f * (F_I_des - F_I_est)
        delta_roll = np.clip(delta_roll, -0.05, 0.05)

        return F_I_des, delta_roll, self.delta, fc, self.active


# ================================================================
# 双臂 PyBullet 模型
# ================================================================
class DualArmPyBullet:
    def __init__(self, urdf_path):
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(os.getcwd(), urdf_path)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.all_revolute = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.all_revolute.append(info[0])
        self.n_total = len(self.all_revolute)
        self.joint_to_pos = {jid: idx for idx, jid in enumerate(self.all_revolute)}
        self.left_joints = []
        self.right_joints = []
        for jid in self.all_revolute:
            name = p.getJointInfo(self.robot_id, jid)[1].decode().lower()
            if name.startswith("left_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.left_joints.append(jid)
            elif name.startswith("right_") and ("shoulder" in name or "elbow" in name or "wrist" in name):
                self.right_joints.append(jid)
        if len(self.left_joints) != 7:
            self.left_joints = self.all_revolute[:7]
        if len(self.right_joints) != 7:
            self.right_joints = self.all_revolute[-7:]
        self.left_ee = self.left_joints[-1] + 1
        self.right_ee = self.right_joints[-1] + 1
        self.left_pos = [self.joint_to_pos[jid] for jid in self.left_joints]
        self.right_pos = [self.joint_to_pos[jid] for jid in self.right_joints]
        print(f"[PyBullet] 左臂: {self.left_joints}, 右臂: {self.right_joints}")

    def _build_full(self, arm, q7, dq7=None):
        full_q = np.zeros(self.n_total)
        pos = self.left_pos if arm == "left" else self.right_pos
        for i, p_idx in enumerate(pos):
            full_q[p_idx] = q7[i]
        if dq7 is not None:
            full_dq = np.zeros(self.n_total)
            for i, p_idx in enumerate(pos):
                full_dq[p_idx] = dq7[i]
            return full_q, full_dq
        return full_q

    def update_states(self, left_q, right_q):
        for i, jidx in enumerate(self.left_joints):
            p.resetJointState(self.robot_id, jidx, left_q[i])
        for i, jidx in enumerate(self.right_joints):
            p.resetJointState(self.robot_id, jidx, right_q[i])

    def compute_dynamics(self, arm, q, dq):
        full_q, full_dq = self._build_full(arm, q, dq)
        zero = [0.0] * self.n_total
        pos = self.left_pos if arm == "left" else self.right_pos
        G_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), zero, zero))
        Cq_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), full_dq.tolist(), zero)) - G_full
        G = G_full[pos]
        Cq = Cq_full[pos]
        M = []
        for k in range(7):
            aa = np.zeros(self.n_total)
            aa[pos[k]] = 1.0
            T_full = np.array(p.calculateInverseDynamics(self.robot_id, full_q.tolist(), zero, aa.tolist())) - G_full
            M.append(T_full[pos])
        return np.array(M).T, Cq, G

    def compute_jacobian(self, arm, q, dq):
        full_q, full_dq = self._build_full(arm, q, dq)
        ee = self.left_ee if arm == "left" else self.right_ee
        pos = self.left_pos if arm == "left" else self.right_pos
        J_p, _ = p.calculateJacobian(self.robot_id, ee, [0,0,0],
                                       full_q.tolist(), full_dq.tolist(), [0.0]*self.n_total)
        return np.array(J_p)[:, pos]

    def disconnect(self):
        p.disconnect(self.client)


# ================================================================
# 工具函数
# ================================================================
def quintic_interpolate(start, goal, t, duration):
    ratio = np.clip(t / duration, 0.0, 1.0)
    s = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
    return start + (goal - start) * s

def check_joint_limits(q, q_min, q_max):
    return np.clip(q, q_min, q_max)

def check_velocity_limits(dq, dq_max=2.0):
    return np.clip(dq, -dq_max, dq_max)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="DIFR双臂协作搬运实验（力传感器版本，公式41 QP）")
    parser.add_argument('--left_q', type=float, nargs=7, required=True)
    parser.add_argument('--right_q', type=float, nargs=7, required=True)
    parser.add_argument('--move-time', type=float, default=10.0)
    parser.add_argument('--grasp-time', type=float, default=10.0)
    parser.add_argument('--test-time', type=float, default=30.0)
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--gmo-gain', type=float, default=15.0)
    # 力传感器参数
    parser.add_argument('--shm-name', type=str, default='6_axis_force_shm',
                        help='六维力传感器共享内存名称')
    parser.add_argument('--shm-backend', choices=['sysv', 'posix'], default='sysv',
                        help='ATI/SOEM使用sysv；旧POSIX发布端使用posix')
    parser.add_argument('--shm-key', type=lambda x: int(x, 0), default=0x1234,
                        help='ATI FT_api.h中的System V key，默认0x1234')
    parser.add_argument('--force-bias-samples', type=int, default=200,
                        help='力传感器零偏采集样本数')
    parser.add_argument('--normal-axis', choices='xyz', default='z')
    parser.add_argument('--tangent-axis', choices='xyz', default='y')
    parser.add_argument('--normal-sign', type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument('--tangent-sign', type=float, choices=[-1.0, 1.0], default=1.0)
    # DIFR QP参数
    parser.add_argument('--F0', type=float, default=16.35,
                        help='标称期望内力F_0 (N), 默认16.35=1kg*9.81/(2*0.3)')
    parser.add_argument('--F-min', type=float, default=5.0)
    parser.add_argument('--F-max', type=float, default=25.0)
    parser.add_argument('--mu', type=float, default=0.3)
    parser.add_argument('--m-object', type=float, default=1.0)
    parser.add_argument('--alpha0', type=float, default=1.0)
    parser.add_argument('--alpha1', type=float, default=1e8,
                        help='滑移代价权重；因delta单位为m，通常需较大数值')
    parser.add_argument('--delta-min', type=float, default=-0.05)
    parser.add_argument('--delta-max', type=float, default=0.05)
    parser.add_argument('--fc-threshold', type=float, default=0.8)
    parser.add_argument('--collision-threshold', type=float, default=8.0)
    parser.add_argument('--K-f', type=float, default=0.002,
                        help='内力→肩roll位置增益')
    parser.add_argument('--roll-action-sign', type=float, choices=[-1.0, 1.0], default=1.0,
                        help='若增大指令反而减小夹持力，设为-1')
    # 关节角增量限幅
    parser.add_argument('--joint-delta-max', type=float, default=0.008,
                        help='每帧关节角最大变化量(rad)')
    parser.add_argument('--output', type=str, default='difr_force.txt')
    parser.add_argument('--urdf', type=str, default='description/g1_14dof_brainco_hand.urdf')
    args = parser.parse_args()

    total_time = args.move_time + args.grasp_time + args.test_time
    print("=" * 80)
    print("  DIFR 双臂协作搬运实验 (力传感器版本, Eq.41 QP)")
    print("=" * 80)
    print(f"  阶段1 MOVE:  {args.move_time}s")
    print(f"  阶段2 GRASP: {args.grasp_time}s (GMO去偏采集)")
    print(f"  阶段3 TEST:  {args.test_time}s (DIFR激活)")
    print(f"  力传感器共享内存: {args.shm_name}")
    print(f"  QP参数: F_0={args.F0}, F∈[{args.F_min},{args.F_max}], μ={args.mu}, m_o={args.m_object}")
    print("=" * 80)

    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"difr_force_experiment_{ts}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    q_min = np.array([-2.0, -1.2, -2.5, -0.7, -1.6, -1.6, -1.2])
    q_max = np.array([ 2.0,  1.2,  2.5,  1.8,  1.6,  1.6,  1.2])
    dq_max = 0.2

    # 初始化力传感器
    print(f"\n[力传感器] 连接共享内存: {args.shm_name}")
    force_reader = ForceSensorReader(args.shm_name, data_size=6,
                                     backend=args.shm_backend,
                                     shm_key=args.shm_key)
    if not force_reader.connect():
        raise RuntimeError("左手六维力传感器共享内存未就绪，实验未启动。")
    force_data = ForceData(args.normal_axis, args.tangent_axis,
                           args.normal_sign, args.tangent_sign)

    pb = DualArmPyBullet(args.urdf)
    gmo_left = GMOObserver(7, args.gmo_gain)
    gmo_right = GMOObserver(7, args.gmo_gain)
    difr = DIFRController(F_0=args.F0, F_min=args.F_min, F_max=args.F_max,
                           mu=args.mu, m_object=args.m_object,
                           alpha0=args.alpha0, alpha1=args.alpha1,
                           delta_min=args.delta_min, delta_max=args.delta_max,
                           fc_threshold=args.fc_threshold,
                           collision_threshold=args.collision_threshold,
                           K_f=args.K_f)

    print(f"\n[初始化] 连接 G1...")
    server = G1Server(network_interface=args.interface, shm_name=args.shm_name)
    server.start()
    time.sleep(0.5)
    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    # 力传感器零偏采集（在MOVE之前，静止状态下）
    print(f"\n[力传感器] 采集零偏 ({args.force_bias_samples}样本)...")
    bias_samples_l = []
    for i in range(args.force_bias_samples):
        raw = force_reader.read_force()
        bias_samples_l.append(raw[0:6].copy())
        time.sleep(0.01)
    force_data.left_bias = np.mean(bias_samples_l, axis=0)
    force_data.bias_applied = True
    print(f"[力传感器] 左手零偏: {force_data.left_bias}")

    data_file = open(output_path, 'w')
    data_file.write("# DIFR双臂协作搬运实验数据（力传感器版本，公式41 QP）\n")
    data_file.write(f"# 方法: DIFR (Dynamic Internal Force Regulation)\n")
    data_file.write(f"# 力传感器: 共享内存 {args.shm_name}\n")
    data_file.write(f"# 时间: {datetime.now()}\n")
    data_file.write(f"# 阶段: MOVE={args.move_time}s, GRASP={args.grasp_time}s, TEST={args.test_time}s\n")
    data_file.write(f"# QP参数: F_0={args.F0}, F_min={args.F_min}, F_max={args.F_max}, mu={args.mu}, m_object={args.m_object}\n")
    data_file.write("# 列: time phase(0=MOVE,1=GRASP,2=TEST) "
                     "left_q(7) right_q(7) "
                     "left_gmo_F(3) right_gmo_F(3) "
                     "force_left_normal "
                     "F_I_est(内力估计,N) F_E_gmo(GMO外力,N) F_E_sensor(传感器外力,N) "
                     "F_I_des(期望内力) delta(滑动位移) delta_roll(肩roll偏移) "
                     "fc(摩擦锥裕度) difr_active "
                     "force_left_6dof(6)\n")
    data_file.write("#" + "=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("  实验开始")
    print("=" * 80)

    t_step = 0.01
    exp_start = time.time()
    last_print = 0.0
    record_count = 0
    last_time = time.time()
    gmo_left.reset()
    gmo_right.reset()
    difr.reset()

    # 关节角增量限幅
    last_target_l = start_l.copy()
    last_target_r = start_r.copy()

    # GMO稳态偏置（GRASP阶段最后2秒采集，TEST阶段去除）
    gmo_bias_F_l = np.zeros(3)
    gmo_bias_F_r = np.zeros(3)
    gmo_bias_samples_l = []
    gmo_bias_samples_r = []
    gmo_bias_applied = False
    bias_collect_start = args.move_time + args.grasp_time - 2.0

    def get_phase(exp_time):
        if exp_time < args.move_time:
            return 0
        elif exp_time < args.move_time + args.grasp_time:
            return 1
        else:
            return 2

    def run_one_step(exp_time, target_l, target_r):
        nonlocal last_time, last_print, record_count
        nonlocal gmo_bias_F_l, gmo_bias_F_r, gmo_bias_samples_l, gmo_bias_samples_r, gmo_bias_applied
        nonlocal last_target_l, last_target_r

        now = time.time()
        dt = now - last_time
        last_time = now
        phase = get_phase(exp_time)

        # 读取力传感器
        raw_force = force_reader.read_force()
        force_data.update(raw_force)
        f_left_6d = force_data.get_forces()
        F_I_est, f_left_normal = force_data.get_internal_force()
        F_E_sensor = force_data.get_external_force_from_sensor()

        # 关节状态
        cur = server.manager.get_current_arm_states()
        cur_l_q = np.array(cur["left_q"])
        cur_r_q = np.array(cur["right_q"])
        cur_l_dq = np.array(cur["left_dq"])
        cur_r_dq = np.array(cur["right_dq"])
        cur_l_tau = np.array(cur["left_tau"])
        cur_r_tau = np.array(cur["right_tau"])
        cur_l_dq = check_velocity_limits(cur_l_dq, dq_max)
        cur_r_dq = check_velocity_limits(cur_r_dq, dq_max)

        pb.update_states(cur_l_q, cur_r_q)

        # 左臂 GMO
        M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
        r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)
        J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
        F_l = np.linalg.pinv(J_l.T) @ r_l

        # 右臂 GMO
        M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
        r_r = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
        J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
        F_r = np.linalg.pinv(J_r.T) @ r_r

        # GMO稳态偏置处理
        if phase == 1 and exp_time >= bias_collect_start:
            gmo_bias_samples_l.append(F_l.copy())
            gmo_bias_samples_r.append(F_r.copy())
        elif phase == 2:
            if not gmo_bias_applied and len(gmo_bias_samples_l) > 10:
                gmo_bias_F_l = np.mean(gmo_bias_samples_l, axis=0)
                gmo_bias_F_r = np.mean(gmo_bias_samples_r, axis=0)
                gmo_bias_applied = True
                print(f"\n[GMO] 稳态偏置: L={gmo_bias_F_l}, R={gmo_bias_F_r}")
            if gmo_bias_applied:
                F_l = F_l - gmo_bias_F_l
                F_r = F_r - gmo_bias_F_r

        # GMO碰撞力（双臂平均）
        gmo_force = (np.linalg.norm(F_l) + np.linalg.norm(F_r)) / 2.0

        # 论文公式(39)-(41)所需接触力直接取左手六维传感器。
        F_E = F_E_sensor

        # DIFR控制器（阶段2和阶段3生效）
        if phase >= 1:
            F_I_des, delta_roll, delta, fc, difr_active = difr.update(
                F_E, F_I_est, gmo_force, dt)
            target_l[1] -= args.roll_action_sign * delta_roll / 2.0
            target_r[1] += args.roll_action_sign * delta_roll / 2.0
        else:
            F_I_des = args.F0
            delta_roll = 0.0
            delta = 0.0
            fc = 0.0
            difr_active = False

        # 关节角限位
        target_l = check_joint_limits(target_l, q_min, q_max)
        target_r = check_joint_limits(target_r, q_min, q_max)

        # 关节角增量限幅
        delta_l = target_l - last_target_l
        delta_r = target_r - last_target_r
        delta_l = np.clip(delta_l, -args.joint_delta_max, args.joint_delta_max)
        delta_r = np.clip(delta_r, -args.joint_delta_max, args.joint_delta_max)
        target_l = last_target_l + delta_l
        target_r = last_target_r + delta_r
        last_target_l = target_l.copy()
        last_target_r = target_r.copy()

        # 发送控制指令
        if phase != 2:
            server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
                                        [0.0]*7, [0.0]*7)
        else:
            print(f"\n[阶段{phase}] 期望内力 F_I_des={F_I_des:.2f}N, 实际内力 F_I_est={F_I_est:.2f}N, 滑动位移 δ={delta:.4f}, 摩擦裕度 fc={fc:.2f}")
            print(f"\n夹持目标姿态: 左臂 {target_l}, 右臂 {target_r}")        

        # 记录
        row = np.concatenate([
            [exp_time, phase],
            cur_l_q, cur_r_q,
            F_l, F_r,
            [f_left_normal],
            [F_I_est, F_E, F_E_sensor],
            [F_I_des, delta, delta_roll, fc, 1.0 if difr_active else 0.0],
            f_left_6d
        ])
        data_file.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        record_count += 1

        # 打印
        if exp_time - last_print >= 0.1:
            phase_str = ["MOVE", "GRASP", "TEST"][phase]
            active_str = "DIFR-ON" if difr_active else "difr-off"
            fc_str = f"fc={fc:.2f}" + ("!" if fc > 1.0 else "")
            print(f"\r  t={exp_time:5.2f}s [{phase_str}] [{active_str}] | "
                  f"GMO={gmo_force:5.1f}N F_E={F_E:5.1f}N | "
                  f"F_I_des={F_I_des:5.1f} est={F_I_est:5.1f} | "
                  f"δ={delta:+.4f} {fc_str} | "
                  f"delta_roll={delta_roll:+.4f} | "
                  f"左传感器 Fn={f_left_normal:5.1f} Ft={F_E_sensor:+5.1f}",
                  end="", flush=True)
            last_print = exp_time

    try:
        # 阶段1: MOVE
        print(f"\n[阶段1] 运动到预备姿态 ({args.move_time}s)...")
        move_start = time.time()
        while True:
            elapsed = time.time() - move_start
            exp_time = time.time() - exp_start
            if elapsed >= args.move_time:
                break
            target_l = quintic_interpolate(start_l, goal_l, elapsed, args.move_time)
            target_r = quintic_interpolate(start_r, goal_r, elapsed, args.move_time)
            run_one_step(exp_time, target_l.copy(), target_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段1] 到达预备姿态")

        # 阶段2: GRASP
        print(f"\n[阶段2] 保持姿态，GMO去偏采集 ({args.grasp_time}s)...")
        grasp_start = time.time()
        while True:
            elapsed = time.time() - grasp_start
            exp_time = time.time() - exp_start
            if elapsed >= args.grasp_time:
                break
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段2] 完成")

        # 阶段3: TEST
        print(f"\n[阶段3] 碰撞测试 ({args.test_time}s) — 可施加外部冲击，DIFR将激活")
        test_start = time.time()
        while True:
            elapsed = time.time() - test_start
            exp_time = time.time() - exp_start
            if elapsed >= args.test_time:
                break
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段3] 测试完成")

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断")
    except Exception as e:
        print(f"\n\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 80)
        print("  实验结束，清理中...")
        print("=" * 80)

        data_file.write(f"# 记录总帧数: {record_count}\n")
        data_file.close()
        print(f"[数据] 已保存 {record_count} 帧到: {output_path}")

        try:
            cur = server.manager.get_current_arm_states()
            # server.manager.set_arm_poses(cur["left_q"], cur["right_q"], [0.0]*7, [0.0]*7)
            server.stop()
            print("[G1] 已停止")
        except Exception as e:
            print(f"[G1] 停止异常: {e}")

        force_reader.close()
        print("[力传感器] 已断开")

        pb.disconnect()
        print("[PyBullet] 已断开")
        print(f"\n[完成] 数据文件: {output_path}")


if __name__ == "__main__":
    main()
