#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-03 (v2: 三阶段实验)
Description: 双臂协作搬运实验 — SIFR (Static Internal Force Regulation)
             对比方法：期望内力固定为常数，不随外部状态变化

实验三阶段（与DIFR完全一致，便于对比）：
  阶段1 MOVE  (0~10s): 双臂五次多项式运动到搬运预备姿态
  阶段2 GRASP (10~20s): 夹持物体，跟踪固定期望内力F_fixed
  阶段3 TEST  (20~50s): 碰撞测试，期望内力保持固定（无动态调节）

SIFR vs DIFR 的唯一区别：
  SIFR: F_I_des = F_fixed (常数)
  DIFR: F_I_des = QP优化结果(公式41)，随滑动动力学动态变化

用法：
  python sifr_experiment.py \
      --left_q  -0.8 0.7 -0.7 0.4 0.0 -0.6 0.0 \
      --right_q  0.8 0.7  0.7 0.4 0.0  0.6 0.0 \
      --F-fixed 15.0
'''
import os
import sys
import time
import argparse
import threading
import multiprocessing as mp
import numpy as np
import pybullet as p
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import G1Server


# ================================================================
# 触觉+灵巧手子进程（spawn 隔离 DDS）
# ================================================================
def tactile_subscriber_worker(data_queue, log_queue, cmd_queue, stop_event):
    try:
        import rclpy
        from rclpy.node import Node
        from ros2_stark_msgs.msg import TouchStatus
        from hand_controller import HandController
    except ImportError as e:
        log_queue.put(f"[触觉] 导入失败: {e}")
        return
    FINGER_COUNT = 5

    class TactileHandNode(Node):
        def __init__(self):
            super().__init__("sifr_tactile_hand")
            self.left_sub = self.create_subscription(
                TouchStatus, "/left_hand/touch_status_126", self.left_cb, 10)
            self.right_sub = self.create_subscription(
                TouchStatus, "/right_hand/touch_status_127", self.right_cb, 10)
            self.hand = HandController(self)
            log_queue.put("[触觉+手控] 订阅和控制器已创建")

        def _extract(self, msg):
            nf1 = np.zeros(FINGER_COUNT)
            for i in range(min(FINGER_COUNT, len(msg.data))):
                nf1[i] = float(msg.data[i].normal_force1)
            return nf1

        def left_cb(self, msg):
            data_queue.put(("left", self._extract(msg).tolist(), time.time()))

        def right_cb(self, msg):
            data_queue.put(("right", self._extract(msg).tolist(), time.time()))

        def execute_cmd(self, cmd):
            try:
                action = cmd[0]
                if action == 'grasp':
                    self.hand.grasp(cmd[1], cmd[2], cmd[3])
                    log_queue.put(f"[手控] {cmd[1]}手执行姿态: {cmd[2]}")
                elif action == 'release':
                    self.hand.release(cmd[1], cmd[2])
                    log_queue.put(f"[手控] {cmd[1]}手松开")
                elif action == 'set_positions':
                    self.hand.set_positions(cmd[1], cmd[2], mode=5, durations=[cmd[3]]*6)
                    log_queue.put(f"[手控] {cmd[1]}手设置位置: {cmd[2]}")
                elif action == 'set_single':
                    self.hand.set_single(cmd[1], cmd[2], cmd[3], duration=cmd[4])
                    log_queue.put(f"[手控] {cmd[1]}手电机{cmd[2]}→{cmd[3]}")
                elif action == 'both_grasp':
                    self.hand.both_grasp(cmd[1], cmd[2])
                    log_queue.put(f"[手控] 双手执行姿态: {cmd[1]}")
                elif action == 'both_release':
                    self.hand.both_release(cmd[1])
                    log_queue.put(f"[手控] 双手松开")
            except Exception as e:
                log_queue.put(f"[手控] 指令执行失败: {e}")

    try:
        rclpy.init()
        node = TactileHandNode()
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
            while not cmd_queue.empty():
                try:
                    node.execute_cmd(cmd_queue.get_nowait())
                except Exception:
                    break
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        log_queue.put(f"[触觉] 错误: {e}")


# ================================================================
# 触觉数据容器
# ================================================================
class TactileData:
    def __init__(self, scale=0.01):
        self.left_nf1 = np.zeros(5)   # raw值（不做转换）
        self.right_nf1 = np.zeros(5)  # raw值
        self.scale = scale              # raw → N 转换系数（需砝码实验标定）
        self._lock = threading.Lock()
    def update(self, hand, nf1):
        with self._lock:
            if hand == "left":
                self.left_nf1 = np.array(nf1, dtype=float)
            else:
                self.right_nf1 = np.array(nf1, dtype=float)
    def get_sums_N(self):
        '''标定后的法向力总和（单位N）= raw_sum × scale'''
        with self._lock:
            return float(np.sum(self.left_nf1) * self.scale), float(np.sum(self.right_nf1) * self.scale)
    def get_sums_raw(self):
        '''raw值（用于调试）'''
        with self._lock:
            return float(np.sum(self.left_nf1)), float(np.sum(self.right_nf1))
    def get_fingers_raw(self):
        with self._lock:
            return self.left_nf1.copy(), self.right_nf1.copy()


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
# 手自重补偿器
# ================================================================
class HandGravityCompensator:
    def __init__(self, hand_mass=0.45, g=9.81):
        self.F_hand = np.array([0.0, 0.0, -hand_mass * g])
    def compensate(self, r_gmo, J_p):
        tau_hand = J_p.T @ self.F_hand
        return r_gmo - tau_hand, tau_hand


# ================================================================
# SIFR 控制器（固定期望内力）
# ================================================================
class SIFRController:
    '''
    Static Internal Force Regulation — 期望内力固定为常数
    与DIFR的唯一区别：F_I_des = F_fixed，不随滑动状态变化
    仍记录滑动位移δ用于对比分析（但不用于调节）
    '''
    def __init__(self, F_fixed=15.0, mu=0.3, m_object=1.0, K_f=0.002):
        self.F_fixed = F_fixed
        self.mu = mu
        self.m_object = m_object
        self.K_f = K_f
        # 仅用于记录对比，不参与控制
        self.delta = 0.0
        self.delta_dot = 0.0
        self.F_I_des = F_fixed
        self.active = False  # SIFR永远不"激活"动态调节

    def reset(self):
        self.delta = 0.0
        self.delta_dot = 0.0
        self.F_I_des = self.F_fixed

    def update(self, F_E, F_I_est, gmo_force_norm, dt):
        '''
        SIFR更新：期望内力固定，仅记录滑动状态和摩擦锥裕度用于对比
        Returns:
            F_I_des: 固定期望内力
            delta_roll: 肩关节roll偏移量（仍跟踪固定内力）
            delta: 滑动位移估计（仅记录）
            fc: 摩擦锥裕度 = F_E / (mu * F_I_des)，fc>1表示即将滑动
            active: 始终False（无动态调节）
        '''
        # 期望内力固定
        F_I_des = self.F_fixed

        # 摩擦锥裕度（仅记录，不用于调节）
        fc = F_E / max(self.mu * F_I_des, 1e-6)

        # 仅记录滑动状态（公式39-40），不用于调节
        self.delta_dot = self.delta_dot + (F_E - self.mu * F_I_des) / self.m_object * dt
        self.delta = self.delta + self.delta_dot * dt
        # 简单限幅防止数值发散
        self.delta = np.clip(self.delta, -0.5, 0.5)
        self.delta_dot = np.clip(self.delta_dot, -1.0, 1.0)

        self.F_I_des = F_I_des
        self.active = False

        # 位置控制下的内力跟踪（与DIFR相同的实现）
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

def drain_queue(queue):
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except Exception:
            break
    return items

def check_joint_limits(q, q_min, q_max):
    return np.clip(q, q_min, q_max)

def check_velocity_limits(dq, dq_max=2.0):
    return np.clip(dq, -dq_max, dq_max)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="SIFR双臂协作搬运实验（固定期望内力）")
    parser.add_argument('--left_q', type=float, nargs=7, required=True)
    parser.add_argument('--right_q', type=float, nargs=7, required=True)
    parser.add_argument('--move-time', type=float, default=10.0)
    parser.add_argument('--grasp-time', type=float, default=10.0)
    parser.add_argument('--test-time', type=float, default=30.0)
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--gmo-gain', type=float, default=5.0)
    parser.add_argument('--hand-mass', type=float, default=0.45)
    parser.add_argument('--tactile-scale', type=float, default=0.01,
                        help='触觉raw→N转换系数 (N/raw), 需砝码实验标定. 默认0.01即100raw=1N')
    # 灵巧手控制参数
    parser.add_argument('--hand-enable', action='store_true', default=False,
                        help='启用灵巧手控制（默认禁用）')
    parser.add_argument('--hand-pose', type=str, default='grasp',
                        choices=['open', 'grasp', 'power', 'pinch', 'tripod'],
                        help='GRASP阶段灵巧手预设姿态')
    parser.add_argument('--hand-duration', type=int, default=1500,
                        help='灵巧手运动时间(ms)')
    parser.add_argument('--hand-left-pos', type=int, nargs=6, default=None,
                        help='左手6电机目标位置, 覆盖--hand-pose')
    parser.add_argument('--hand-right-pos', type=int, nargs=6, default=None,
                        help='右手6电机目标位置, 覆盖--hand-pose')
    # SIFR参数（物理标定：1kg物体，μ=0.3 → F_fixed = m*g/(2*μ) ≈ 16.35N/臂）
    parser.add_argument('--F-fixed', type=float, default=16.35, help='固定期望内力 (N/臂), 默认16.35=1kg*9.81/(2*0.3)')
    parser.add_argument('--mu', type=float, default=0.3, help='摩擦系数(仅用于记录滑动和fc)')
    parser.add_argument('--m-object', type=float, default=1.0, help='物体质量(仅用于记录滑动)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--urdf', type=str, default='description/g1_14dof_brainco_hand.urdf')
    args = parser.parse_args()

    total_time = args.move_time + args.grasp_time + args.test_time
    print("=" * 80)
    print("  SIFR 双臂协作搬运实验 (Static Internal Force Regulation)")
    print("=" * 80)
    print(f"  阶段1 MOVE:  {args.move_time}s (运动到预备姿态)")
    print(f"  阶段2 GRASP: {args.grasp_time}s (夹持物体，跟踪F_fixed={args.F_fixed}N)")
    print(f"  阶段3 TEST:  {args.test_time}s (碰撞测试，期望内力保持固定)")
    print(f"  固定内力: F_fixed={args.F_fixed}N")
    print("=" * 80)

    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sifr_experiment_{ts}.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    print(f"\n[数据] 记录文件: {output_path}")

    q_min = np.array([-2.0, -1.2, -2.5, -0.7, -1.6, -1.4, -1.2])
    q_max = np.array([ 2.0,  2.0,  2.5,  1.8,  1.6,  1.4,  1.2])
    dq_max = 0.2

    pb = DualArmPyBullet(args.urdf)
    gmo_left = GMOObserver(7, args.gmo_gain)
    gmo_right = GMOObserver(7, args.gmo_gain)
    hand_comp_left = HandGravityCompensator(args.hand_mass)
    hand_comp_right = HandGravityCompensator(args.hand_mass)
    sifr = SIFRController(F_fixed=args.F_fixed, mu=args.mu, m_object=args.m_object)

    print(f"\n[初始化] 连接 G1...")
    server = G1Server(network_interface=args.interface, shm_name="dev/shm/6_axis_force_shm")
    server.start()
    time.sleep(0.5)
    current = server.manager.get_current_arm_states()
    start_l = np.array(current["left_q"])
    start_r = np.array(current["right_q"])
    goal_l = np.array(args.left_q)
    goal_r = np.array(args.right_q)
    print(f"[初始化] 当前左臂: {start_l}")
    print(f"[初始化] 当前右臂: {start_r}")

    tactile_data = TactileData(scale=args.tactile_scale)
    ctx = mp.get_context('spawn')
    data_queue = ctx.Queue(maxsize=200)
    log_queue = ctx.Queue(maxsize=50)
    cmd_queue = ctx.Queue(maxsize=50)
    stop_event = ctx.Event()
    tactile_process = ctx.Process(target=tactile_subscriber_worker,
                                   args=(data_queue, log_queue, cmd_queue, stop_event),
                                   daemon=True)
    tactile_process.start()
    print(f"[触觉] 子进程 PID: {tactile_process.pid}")
    if args.hand_enable:
        print(f"[手控] 灵巧手控制已启用, GRASP姿态: {args.hand_pose}")
    time.sleep(3.0)
    for msg in drain_queue(log_queue):
        print(f"  {msg}")

    data_file = open(output_path, 'w')
    data_file.write("# SIFR双臂协作搬运实验数据（固定期望内力）\n")
    data_file.write(f"# 方法: SIFR (Static Internal Force Regulation)\n")
    data_file.write(f"# 时间: {datetime.now()}\n")
    data_file.write(f"# 阶段: MOVE={args.move_time}s, GRASP={args.grasp_time}s, TEST={args.test_time}s\n")
    data_file.write(f"# 固定内力: F_fixed={args.F_fixed}\n")
    data_file.write("# 列: time phase(0=MOVE,1=GRASP,2=TEST) "
                     "left_q(7) right_q(7) left_dq(7) right_dq(7) "
                     "left_gmo_F(3) right_gmo_F(3) "
                     "tactile_left tactile_right "
                     "F_E(估计外力) F_I_des(期望内力=固定) F_I_est(实际内力) "
                     "delta(滑动位移,仅记录) delta_roll(肩roll偏移) fc(摩擦锥裕度) difr_active(始终0) "
                     "left_tactile_5f(5) right_tactile_5f(5)\n")
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
    sifr.reset()

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

        now = time.time()
        dt = now - last_time
        last_time = now
        phase = get_phase(exp_time)

        for item in drain_queue(data_queue):
            hand, nf1, ts = item
            tactile_data.update(hand, nf1)
        for msg in drain_queue(log_queue):
            print(f"\n  [触觉] {msg}")

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

        # 左臂 GMO + 手补偿
        if phase == 0:
            M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
            r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)
            J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
            F_l = np.linalg.pinv(J_l.T) @ r_l
        else:
            M_l, Cq_l, G_l = pb.compute_dynamics("left", cur_l_q, cur_l_dq)
            r_l = gmo_left.update(M_l, Cq_l, G_l, cur_l_tau, cur_l_dq, dt)

            J_l = pb.compute_jacobian("left", cur_l_q, cur_l_dq)
            r_l, _ = hand_comp_left.compensate(r_l, J_l)
            F_l = np.linalg.pinv(J_l.T) @ r_l

        # 右臂 GMO + 手补偿
        if phase == 0:
            M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
            r_r = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
            J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
            F_r = np.linalg.pinv(J_r.T) @ r_r
        else:
            M_r, Cq_r, G_r = pb.compute_dynamics("right", cur_r_q, cur_r_dq)
            r_r_raw = gmo_right.update(M_r, Cq_r, G_r, cur_r_tau, cur_r_dq, dt)
            J_r = pb.compute_jacobian("right", cur_r_q, cur_r_dq)
            r_r, _ = hand_comp_right.compensate(r_r_raw, J_r)
            F_r = np.linalg.pinv(J_r.T) @ r_r

        # 触觉力（标定后单位N）
        t_left_N, t_right_N = tactile_data.get_sums_N()
        t_left_raw, t_right_raw = tactile_data.get_sums_raw()
        f_left_5, f_right_5 = tactile_data.get_fingers_raw()

        # ===== GMO稳态偏置处理 =====
        if phase == 1 and exp_time >= bias_collect_start:
            gmo_bias_samples_l.append(F_l.copy())
            gmo_bias_samples_r.append(F_r.copy())
        elif phase == 2:
            if not gmo_bias_applied and len(gmo_bias_samples_l) > 10:
                gmo_bias_F_l = np.mean(gmo_bias_samples_l, axis=0)
                gmo_bias_F_r = np.mean(gmo_bias_samples_r, axis=0)
                gmo_bias_applied = True
                print(f"\n[GMO] 稳态偏置已采集: L={gmo_bias_F_l}, R={gmo_bias_F_r}")
            if gmo_bias_applied:
                F_l = F_l - gmo_bias_F_l
                F_r = F_r - gmo_bias_F_r

        gmo_force = (np.linalg.norm(F_l) + np.linalg.norm(F_r)) / 2.0
        F_E = np.sqrt(F_l[0]**2 + F_l[1]**2 + F_r[0]**2 + F_r[1]**2) / 2.0
        F_I_est = abs(t_left_N - t_right_N) * 0.5

        # SIFR控制器（阶段2和阶段3生效）
        if phase >= 1:
            F_I_des, delta_roll, delta, fc, active = sifr.update(F_E, F_I_est, gmo_force, dt)
            target_l[1] += delta_roll / 2.0
            target_r[1] -= delta_roll / 2.0
        else:
            F_I_des = args.F_fixed
            delta_roll = 0.0
            delta = 0.0
            fc = 0.0
            active = False

        target_l = check_joint_limits(target_l, q_min, q_max)
        target_r = check_joint_limits(target_r, q_min, q_max)

        if phase == 0:
            server.manager.set_arm_poses(target_l.tolist(), target_r.tolist(),
                                        [0.0]*7, [0.0]*7)
        else:
            print(f"\n[阶段{phase}] 期望内力 F_I_des={F_I_des:.2f}N, 实际内力 F_I_est={F_I_est:.2f}N, 滑动位移 δ={delta:.4f}, 摩擦裕度 fc={fc:.2f}")
            print(f"\n夹持目标姿态: 左臂 {target_l}, 右臂 {target_r}")

        row = np.concatenate([
            [exp_time, phase],
            cur_l_q, cur_r_q,
            cur_l_dq, cur_r_dq,
            F_l, F_r,
            [t_left_N, t_right_N],
            [F_E, F_I_des, F_I_est],
            [delta, delta_roll, fc, 1.0 if active else 0.0],
            f_left_5, f_right_5
        ])
        data_file.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        record_count += 1

        if exp_time - last_print >= 0.1:
            phase_str = ["MOVE", "GRASP", "TEST"][phase]
            fc_str = f"fc={fc:.2f}" + ("!" if fc > 1.0 else "")
            print(f"\r  t={exp_time:5.2f}s [{phase_str}] [SIFR-fixed] | "
                  f"GMO={gmo_force:5.1f}N F_E={F_E:5.1f}N | "
                  f"F_I={F_I_des:5.1f}(fixed) est={F_I_est:5.1f} | "
                  f"δ={delta:+.4f} {fc_str} | "
                  f"触觉 L={t_left_N:6.2f}N R={t_right_N:6.2f}N",
                  end="", flush=True)
            last_print = exp_time

    try:
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

        print(f"\n[阶段2] 夹持物体，跟踪固定内力 F_fixed={args.F_fixed}N ({args.grasp_time}s)...")

        # 灵巧手夹持控制
        if args.hand_enable:
            if args.hand_left_pos is not None:
                cmd_queue.put(('set_positions', 'left', args.hand_left_pos, args.hand_duration))
            else:
                cmd_queue.put(('grasp', 'left', args.hand_pose, args.hand_duration))
            if args.hand_right_pos is not None:
                cmd_queue.put(('set_positions', 'right', args.hand_right_pos, args.hand_duration))
            else:
                cmd_queue.put(('grasp', 'right', args.hand_pose, args.hand_duration))
            print(f"[手控] 已发送夹持指令: pose={args.hand_pose}, duration={args.hand_duration}ms")
            time.sleep(min(args.hand_duration / 1000.0 + 0.5, 2.0))

        grasp_start = time.time()
        while True:
            elapsed = time.time() - grasp_start
            exp_time = time.time() - exp_start
            if elapsed >= args.grasp_time:
                break
            run_one_step(exp_time, goal_l.copy(), goal_r.copy())
            time.sleep(t_step)
        print(f"\n[阶段2] 夹持完成")

        print(f"\n[阶段3] 碰撞测试 ({args.test_time}s) — 可施加外部冲击，期望内力保持固定")
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

        if tactile_process.is_alive():
            if args.hand_enable:
                try:
                    cmd_queue.put(('both_release', 1000))
                    time.sleep(1.2)
                    print("[手控] 灵巧手已松开")
                except Exception:
                    pass
            stop_event.set()
            tactile_process.join(timeout=5.0)
            if tactile_process.is_alive():
                tactile_process.terminate()
            print("[触觉] 子进程已停止")

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

        pb.disconnect()
        print("[PyBullet] 已断开")
        print(f"\n[完成] 数据文件: {output_path}")


if __name__ == "__main__":
    main()
