"""Precomputed symmetric normal-direction IK actuator for real-time force loops."""
import numpy as np
import pybullet as p


class SymmetricNormalIKActuator:
    """Solve one maximum-closure IK endpoint, then interpolate online."""

    def __init__(self, model, goal_left, goal_right, max_displacement=0.015,
                 damping=0.04, iterations=160, max_step=0.005,
                 q_min=None, q_max=None):
        if max_displacement <= 0:
            raise ValueError("ik_max_displacement must be positive")
        self.model = model
        self.goal = {
            "left": np.asarray(goal_left, float).copy(),
            "right": np.asarray(goal_right, float).copy(),
        }
        self.max_displacement = float(max_displacement)
        self.damping = float(damping)
        self.iterations = int(iterations)
        self.max_step = float(max_step)
        self.q_min = np.asarray(q_min, float)
        self.q_max = np.asarray(q_max, float)
        self.solution, self.metrics = self._solve_endpoint()
        self._print_validation()
        # Do not leave the shared PyBullet model at the IK endpoint.  The
        # hardware loop will overwrite this at its next cycle, but restoring
        # here also makes startup state and diagnostics deterministic.
        self.model.update_states(self.goal["left"], self.goal["right"])

    @staticmethod
    def _quat_error(target, current):
        _, inv = p.invertTransform([0, 0, 0], current)
        _, error = p.multiplyTransforms([0, 0, 0], target, [0, 0, 0], inv)
        axis, angle = p.getAxisAngleFromQuaternion(error)
        if angle > np.pi:
            angle -= 2*np.pi
        return np.asarray(axis) * angle

    def _pose(self, side):
        state = p.getLinkState(
            self.model.robot_id, self.model.left_ee if side == "left" else self.model.right_ee,
            computeForwardKinematics=True)
        return np.asarray(state[4]), np.asarray(state[5])

    def _jacobian6(self, side, q):
        full_q, full_dq = self.model._build_full(side, q, np.zeros(7))
        ee = self.model.left_ee if side == "left" else self.model.right_ee
        pos = self.model.left_pos if side == "left" else self.model.right_pos
        jp, jr = p.calculateJacobian(
            self.model.robot_id, ee, [0, 0, 0], full_q.tolist(),
            full_dq.tolist(), [0.0] * self.model.n_total)
        return np.vstack((np.asarray(jp)[:, pos], np.asarray(jr)[:, pos]))

    def _solve_endpoint(self):
        self.model.update_states(self.goal["left"], self.goal["right"])
        start = {}; quat0 = {}
        for side in ("left", "right"):
            start[side], quat0[side] = self._pose(side)
        normal = start["right"] - start["left"]
        distance = np.linalg.norm(normal)
        if distance < 1e-6:
            raise RuntimeError("left/right EE positions coincide; contact normal is undefined")
        normal /= distance
        target = {
            "left": start["left"] + normal * self.max_displacement,
            "right": start["right"] - normal * self.max_displacement,
        }
        q = {side: self.goal[side].copy() for side in ("left", "right")}

        for _ in range(self.iterations):
            self.model.update_states(q["left"], q["right"])
            for side in ("left", "right"):
                position, quaternion = self._pose(side)
                error = np.r_[target[side] - position,
                              0.4 * self._quat_error(quat0[side], quaternion)]
                jac = self._jacobian6(side, q[side])
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + self.damping**2 * np.eye(6), error)
                q[side] = np.clip(
                    q[side] + np.clip(dq, -self.max_step, self.max_step),
                    self.q_min, self.q_max)

        self.model.update_states(q["left"], q["right"])
        metrics = {"normal": normal}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            position, quaternion = self._pose(side)
            motion = position - start[side]
            normal_motion = sign * float(motion @ normal)
            lateral = np.linalg.norm(motion - sign * normal_motion * normal)
            orientation = np.linalg.norm(self._quat_error(quat0[side], quaternion))
            metrics[side] = {
                "normal_motion": normal_motion,
                "lateral": lateral,
                "orientation": orientation,
            }
        return q, metrics

    def _print_validation(self):
        print(f"[IK] 双手连线法向(world): {self.metrics['normal']}")
        for side in ("left", "right"):
            m = self.metrics[side]
            print(f"[IK] {side}: normal={m['normal_motion']:.6f}m, "
                  f"lateral={m['lateral']:.6f}m, "
                  f"orientation={np.degrees(m['orientation']):.3f}deg")
            print(f"[IK] {side} q solution: {self.solution[side].tolist()}")
        max_lateral = max(self.metrics[s]["lateral"] for s in ("left", "right"))
        max_angle = max(self.metrics[s]["orientation"] for s in ("left", "right"))
        if max_lateral > 0.003 or max_angle > np.radians(3.0):
            raise RuntimeError("IK endpoint validation failed: excessive lateral/orientation error")

        # The real-time loop interpolates joint configurations, so validate
        # that complete path as well as the endpoint.
        self.model.update_states(self.goal["left"], self.goal["right"])
        starts = {}; quats = {}
        for side in ("left", "right"):
            starts[side], quats[side] = self._pose(side)
        path_lateral = 0.0
        path_angle = 0.0
        path_normal_error = 0.0
        normal = self.metrics["normal"]
        for ratio in np.linspace(0.0, 1.0, 11):
            left = self.goal["left"] + ratio * (self.solution["left"] - self.goal["left"])
            right = self.goal["right"] + ratio * (self.solution["right"] - self.goal["right"])
            self.model.update_states(left, right)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                position, quaternion = self._pose(side)
                motion = position - starts[side]
                along = sign * float(motion @ normal)
                lateral = np.linalg.norm(motion - sign * along * normal)
                angle = np.linalg.norm(self._quat_error(quats[side], quaternion))
                path_lateral = max(path_lateral, lateral)
                path_angle = max(path_angle, angle)
                path_normal_error = max(
                    path_normal_error, abs(along - ratio*self.max_displacement))
        print(f"[IK] 插值路径最大误差: normal={path_normal_error:.6f}m, "
              f"lateral={path_lateral:.6f}m, "
              f"orientation={np.degrees(path_angle):.3f}deg")
        if (path_normal_error > 0.003 or path_lateral > 0.003 or
                path_angle > np.radians(3.0)):
            raise RuntimeError("IK interpolation-path validation failed")

    def targets(self, total_closure_command):
        """Map total two-hand closure [m] to symmetric joint targets.

        ``max_displacement`` is the displacement of *each* hand, therefore
        the controller command spans 0 .. 2*max_displacement.
        """
        ratio = np.clip(
            total_closure_command / (2.0 * self.max_displacement), 0.0, 1.0)
        left = self.goal["left"] + ratio * (self.solution["left"] - self.goal["left"])
        right = self.goal["right"] + ratio * (self.solution["right"] - self.goal["right"])
        return left, right, ratio
