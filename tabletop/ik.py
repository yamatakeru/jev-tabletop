# pyright: reportAttributeAccessIssue=false
"""Damped-least-squares IK for the Panda tool centre point (position + orientation)."""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import TCP_OFFSET

ARM_DOF = 7


class IKSolver:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.lo = model.jnt_range[:ARM_DOF, 0]
        self.hi = model.jnt_range[:ARM_DOF, 1]
        # Nullspace posture (the "home" keyframe) keeps the elbow sensible.
        self.q_rest = model.key_qpos[model.key("home").id][:ARM_DOF].copy()

    def pose_error(self, q: np.ndarray, target_pos: np.ndarray) -> float:
        m, d = self.model, self.data
        d.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(m, d)
        R = d.xmat[self.hand].reshape(3, 3)
        tcp = d.xpos[self.hand] + R @ np.array([0, 0, TCP_OFFSET])
        return float(np.linalg.norm(target_pos - tcp))

    def solve_global(self, q_init: np.ndarray, target_pos, target_rot) -> np.ndarray:
        """Multi-seed IK: the current posture can sit in a bad basin (e.g. the parked arm),
        so also start from the rest posture and keep the most accurate solution."""
        best, best_err = None, np.inf
        for seed in (self.q_rest, q_init):
            q = self.solve(seed, target_pos, target_rot, iters=150)
            err = self.pose_error(q, target_pos)
            if err < best_err:
                best, best_err = q, err
        return best

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        iters: int = 60,
        damping: float = 1e-3,
    ) -> np.ndarray:
        m, d = self.model, self.data
        q = np.array(q_init[:ARM_DOF], dtype=float)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err_rot = np.zeros(3)
        q_cur = np.zeros(4)
        q_tgt = np.zeros(4)
        q_err = np.zeros(4)
        q_cur_conj = np.zeros(4)
        mujoco.mju_mat2Quat(q_tgt, target_rot.flatten())
        for _ in range(iters):
            d.qpos[:ARM_DOF] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.hand].reshape(3, 3)
            tcp = d.xpos[self.hand] + R @ np.array([0, 0, TCP_OFFSET])
            err_pos = target_pos - tcp
            mujoco.mju_mat2Quat(q_cur, R.flatten())
            mujoco.mju_negQuat(q_cur_conj, q_cur)
            mujoco.mju_mulQuat(q_err, q_tgt, q_cur_conj)
            mujoco.mju_quat2Vel(err_rot, q_err, 1.0)
            err = np.concatenate([err_pos, err_rot])
            if np.linalg.norm(err_pos) < 5e-4 and np.linalg.norm(err_rot) < 5e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, tcp, self.hand)
            J = np.vstack([jacp, jacr])[:, :ARM_DOF]
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Nullspace pull towards the rest posture.
            N = np.eye(ARM_DOF) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (self.q_rest - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q
