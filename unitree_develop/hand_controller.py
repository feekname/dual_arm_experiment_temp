#!/usr/bin/env python3
'''
Author: pengfei 524560850@qq.com
Date: 2026-09-07
Description: 强脑revo2灵巧手控制器模块

电机映射 (6个电机):
  motor 0: 拇指屈伸 (thumb flexion)
  motor 1: 拇指外展/对掌 (thumb abduction/opposition)
  motor 2: 食指屈伸 (index flexion)
  motor 3: 中指屈伸 (middle flexion)
  motor 4: 无名指屈伸 (ring flexion)
  motor 5: 小指屈伸 (pinky flexion)

控制模式 (mode):
  1: 位置控制
  2: 速度控制
  3: 电流控制 (仅二代手)
  4: PWM控制 (仅二代手)
  5: 位置+期望时间 (平滑运动, durations单位ms)
  6: 位置+期望速度

注意:
  - 位置范围需根据实际手测试确认
  - 从motor_status看, 松开姿态约为 [400, 400, 50, 50, 50, 50]
  - 夹持时四指位置增大(弯曲), 具体范围需测试
  - mode=5时durations字段生效, 可实现平滑运动
'''
import numpy as np

try:
    from ros2_stark_msgs.msg import SetMotorMulti, SetMotorSingle
except ImportError:
    SetMotorMulti = None
    SetMotorSingle = None


# 预设姿态（位置值需根据实际手测试调整）
POSES = {
    'open':      [0, 0,  50,  50,  50,  50],   # 完全张开
    'grasp':     [400, 400, 150, 150, 150, 150],   # 通用夹持（四指半弯）
    'power':     [400, 400, 250, 250, 250, 250],   # 强力夹持（四指全弯）
    'pinch':     [400, 400, 120,  80,  50,  50],   # 指尖捏取（食中指弯曲）
    'tripod':    [400, 400, 150, 150,  50,  50],   # 三指捏取（食中+拇指）
    'max_pos':    [950, 950, 50, 50,  50,  50],   # 最大张开
    'min_pos':    [50, 50, 950, 950,  950,  950],   # 最小抓握
}


class HandController:
    '''
    灵巧手控制器，封装左右手机器人的位置/速度/电流控制
    
    用法:
      hand = HandController(node)  # node是rclpy Node
      hand.grasp('left', pose='grasp', duration=1000)
      hand.set_finger('left', 2, 200)  # 食指单独控制
      hand.release('left', duration=500)
    '''

    def __init__(self, node):
        '''
        Args:
            node: rclpy Node实例，用于创建publisher
        '''
        self.node = node
        self.left_pub = node.create_publisher(
            SetMotorMulti, '/left_hand/set_motor_multi_126', 10)
        self.right_pub = node.create_publisher(
            SetMotorMulti, '/right_hand/set_motor_multi_127', 10)
        self.left_single_pub = node.create_publisher(
            SetMotorSingle, '/left_hand/set_motor_single_126', 10)
        self.right_single_pub = node.create_publisher(
            SetMotorSingle, '/right_hand/set_motor_single_127', 10)

        # 记录当前目标位置（用于触觉引导调节）
        self.current_pos = {'left': POSES['open'].copy(),
                             'right': POSES['open'].copy()}

    def set_positions(self, hand, positions, mode=5, durations=None,
                      speeds=None, currents=None):
        '''
        设置6个电机的目标位置

        Args:
            hand: 'left' or 'right'
            positions: 6个位置值列表 [m0, m1, m2, m3, m4, m5]
            mode: 控制模式 (1=位置, 5=位置+时间, 6=位置+速度)
            durations: 每个电机的期望运动时间(ms), mode=5时生效
            speeds: 每个电机的期望速度, mode=6时生效
            currents: 电流限制
        '''
        msg = SetMotorMulti()
        msg.slave_id = 126 if hand == 'left' else 127
        msg.mode = mode
        msg.positions = [int(p) for p in positions]
        if durations is not None:
            msg.durations = [int(d) for d in durations]
        if speeds is not None:
            msg.speeds = [int(s) for s in speeds]
        if currents is not None:
            msg.currents = [int(c) for c in currents]

        pub = self.left_pub if hand == 'left' else self.right_pub
        pub.publish(msg)
        self.current_pos[hand] = list(positions)

    def set_single(self, hand, motor_id, position, mode=5, duration=500,
                   speed=0, current=0):
        '''
        单独控制一个电机

        Args:
            hand: 'left' or 'right'
            motor_id: 电机编号 0~5
            position: 目标位置
            mode: 控制模式
            duration: 期望时间(ms), mode=5时生效
        '''
        msg = SetMotorSingle()
        msg.slave_id = 126 if hand == 'left' else 127
        msg.mode = mode
        msg.motor_id = motor_id
        msg.position = int(position)
        msg.duration = int(duration)
        msg.speed = int(speed)
        msg.current = int(current)

        pub = self.left_single_pub if hand == 'left' else self.right_single_pub
        pub.publish(msg)
        self.current_pos[hand][motor_id] = position

    def grasp(self, hand, pose='grasp', duration=1000):
        '''
        执行预设夹持姿态

        Args:
            hand: 'left' or 'right'
            pose: 预设姿态名 ('open','grasp','power','pinch','tripod')
            duration: 运动时间(ms)
        '''
        if pose not in POSES:
            print(f"[Hand] 未知姿态: {pose}, 可用: {list(POSES.keys())}")
            return
        positions = POSES[pose].copy()
        self.set_positions(hand, positions, mode=5,
                           durations=[duration] * 6)

    def release(self, hand, duration=1000):
        '''松开手指，回到open姿态'''
        self.grasp(hand, pose='open', duration=duration)

    def both_grasp(self, pose='grasp', duration=1000):
        '''双手同时夹持'''
        self.grasp('left', pose, duration)
        self.grasp('right', pose, duration)

    def both_release(self, duration=1000):
        '''双手同时松开'''
        self.release('left', duration)
        self.release('right', duration)

    def tactile_guided_step(self, hand, tactile_nf1, target_force=50,
                             step=10, min_pos=50, max_pos=300, fingers=None):
        '''
        基于触觉反馈的单步位置调节

        逻辑:
          - 当前触觉力 < target_force → 增大手指弯曲位置(增加夹持)
          - 当前触觉力 > target_force * 1.2 → 减小手指弯曲位置(放松)
          - 中间范围 → 保持不动

        Args:
            hand: 'left' or 'right'
            tactile_nf1: 5指触觉normal_force1数组 (raw值)
            target_force: 目标触觉力 (raw值), 达到此力后停止增加
            step: 每次调节的位置步长
            min_pos: 位置下限(松开)
            max_pos: 位置上限(最大弯曲)
            fingers: 要调节的手指索引列表, 默认[2,3,4,5](四指)

        Returns:
            adjusted: 是否进行了调节
            current_force: 当前触觉力总和
        '''
        if fingers is None:
            fingers = [2, 3, 4, 5]

        current_force = float(np.sum(tactile_nf1))
        positions = self.current_pos[hand].copy()
        adjusted = False

        if current_force < target_force:
            # 力不足，增加弯曲
            for f in fingers:
                positions[f] = min(max_pos, positions[f] + step)
            adjusted = True
        elif current_force > target_force * 1.2:
            # 力过大，放松
            for f in fingers:
                positions[f] = max(min_pos, positions[f] - step)
            adjusted = True

        if adjusted:
            self.set_positions(hand, positions, mode=5, durations=[200] * 6)

        print(f"[Hand] {hand} tactile_force={current_force:.1f}, target={target_force}, adjusted={adjusted}")

        return adjusted, current_force

    def get_current_positions(self, hand):
        '''获取当前目标位置'''
        return self.current_pos[hand].copy()


def test_hand_controller():
    '''测试函数（需在ROS 2环境中运行）'''
    import rclpy
    rclpy.init()
    node = rclpy.create_node('hand_test')
    hand = HandController(node)

    print("测试: 双手张开...")
    hand.both_release(duration=1000)
    rclpy.spin_once(node, timeout_sec=1.0)

    import time
    time.sleep(1.5)

    print("测试: 双手通用夹持...")
    hand.both_grasp(pose='grasp', duration=1000)
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(1.5)

    print("测试: 左手食指单独弯曲到200...")
    hand.set_single('left', 2, 200, duration=500)
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(1.0)

    print("测试: 双手松开...")
    hand.both_release(duration=1000)
    rclpy.spin_once(node, timeout_sec=1.0)

    print("测试：最大位置和最小位置...")
    hand.set_positions('left', POSES['max_pos'], mode=5, durations=[1000]*6)
    hand.set_positions('right', POSES['max_pos'], mode=5, durations=[1000]*6)
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(1.5)

    hand.set_positions('left', POSES['min_pos'], mode=5, durations=[1000]*6)
    hand.set_positions('right', POSES['min_pos'], mode=5, durations=[1000]*6)
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(1.5)

    node.destroy_node()
    rclpy.shutdown()
    print("测试完成")


if __name__ == '__main__':
    test_hand_controller()
