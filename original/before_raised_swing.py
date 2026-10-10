import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


SETTLE_TIME = 0.50
BACKSWING_TIME = 1.00
DOWNSWING_TIME = 0.38
FOLLOW_TIME = 0.70
IMPACT_TIME = SETTLE_TIME + BACKSWING_TIME + DOWNSWING_TIME
FINISH_TIME = IMPACT_TIME + FOLLOW_TIME
FPS = 60
JOINTS = (
    'torso_turn', 'right_shoulder1', 'right_shoulder2',
    'right_wrist_x', 'right_wrist_y', 'left_shoulder1', 'left_shoulder2',
)
KP = np.array([650., 220., 220., 100., 100., 220., 220.])
KD = np.array([70., 28., 28., 12., 12., 28., 28.])


class SwingTrajectory:

    def __init__(self, model):
        self.times = np.array([0., SETTLE_TIME,
                               SETTLE_TIME + BACKSWING_TIME,
                               IMPACT_TIME, FINISH_TIME])
        # Columns: torso, right shoulder 1, wrist x, wrist y (degrees).
        poses = np.array([
            [0., 0., 0., -3.],
            [0., 0., 0., -3.],
            [-50., -50., 0., -45.],
            [8., 0., 0., -3.],
            [40., -30., 0., -45.],
        ])
        speeds = np.zeros_like(poses)
        speeds[3] = [180., 0., 105., 0.]
        self.poses, self.speeds = np.deg2rad(poses), np.deg2rad(speeds)
        # For the original mirrored arms, left angles = -right angles.
        # Use Rodrigues rotations to solve (R1 R2 arm_vector).y = 0.17 m.
        # The analytic solution and implicit derivatives maintain closure
        # without an iterative IK solver or an extra runtime dependency.
        self.axis1 = model.joint('right_shoulder1').axis.copy()
        axis2 = model.joint('right_shoulder2').axis.copy()
        arm = model.body('right_lower_arm').pos + model.body('right_hand').pos
        self.parallel = axis2 * np.dot(axis2, arm)
        self.perpendicular = arm - self.parallel
        self.cross = np.cross(axis2, arm)
        self.lateral = -model.body('right_upper_arm').pos[1]

    def shoulder_curve(self, angle):
        y = np.array([0., 1., 0.])
        n = self.axis1
        cross = np.cross(y, n)
        c, s = np.cos(angle), np.sin(angle)
        row = c*y + s*cross + (1-c)*n[1]*n
        row_d = -s*y + c*cross + s*n[1]*n
        row_dd = -c*y - s*cross + c*n[1]*n
        a, b = row @ self.perpendicular, row @ self.cross
        offset = row @ self.parallel - self.lateral
        alpha = np.arccos(np.clip(-offset/np.hypot(a, b), -1., 1.))
        roots = np.arctan2(b, a) + np.array([-alpha, alpha])
        roots = (roots + np.pi) % (2*np.pi) - np.pi
        other = roots[np.argmin(np.abs(roots))]
        c2, s2 = np.cos(other), np.sin(other)
        arm = c2*self.perpendicular + s2*self.cross + self.parallel
        arm_d = -s2*self.perpendicular + c2*self.cross
        arm_dd = -c2*self.perpendicular - s2*self.cross
        derivative = -(row_d @ arm)/(row @ arm_d)
        second = -(row_dd @ arm + 2*(row_d @ arm_d)*derivative
                   + (row @ arm_dd)*derivative**2)/(row @ arm_d)
        return other, derivative, second

    def sample(self, t):
        t = np.clip(t, self.times[0], self.times[-1])
        i = min(np.searchsorted(self.times, t, side='right')-1, len(self.times)-2)
        duration = self.times[i+1]-self.times[i]
        u = (t-self.times[i])/duration
        p0, p1 = self.poses[i:i+2]
        v0, v1 = self.speeds[i:i+2]*duration
        delta = p1-p0
        c3 = 10*delta-6*v0-4*v1
        c4 = -15*delta+8*v0+7*v1
        c5 = 6*delta-3*v0-3*v1
        p = p0+v0*u+c3*u**3+c4*u**4+c5*u**5
        v = (v0+3*c3*u**2+4*c4*u**3+5*c5*u**4)/duration
        a = (6*c3*u+12*c4*u**2+20*c5*u**3)/duration**2
        s, ds, dds = self.shoulder_curve(p[1])
        sv = ds*v[1]
        sa = dds*v[1]**2+ds*a[1]
        return (np.array([p[0], p[1], s, p[2], p[3], -p[1], -s]),
                np.array([v[0], v[1], sv, v[2], v[3], -v[1], -sv]),
                np.array([a[0], a[1], sa, a[2], a[3], -a[1], -sa]))


class SwingSimulation:
    def __init__(self, *, timestep=None, disturbance_joint=None, disturbance_amplitude=3., disturbance_frequency=8.):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('starting.xml')))
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.trajectory = SwingTrajectory(self.model)
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.motor_ids = np.array([self.model.actuator(f'{n}_motor').id for n in JOINTS])
        self.club_id = self.model.geom('golf_club').id
        self.ball_id = self.model.geom('golf_ball_geom').id
        self.ball_dof = self.model.joint('golf_ball_free').dofadr[0]
        self.inertia = np.empty((self.model.nv, self.model.nv))
        if disturbance_joint is not None and disturbance_joint not in JOINTS:
            raise ValueError(f'Unknown upper-body joint: {disturbance_joint}')
        self.disturbance_joint = disturbance_joint
        self.disturbance_amplitude = disturbance_amplitude
        self.disturbance_frequency = disturbance_frequency
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Start with the original rigid support just touching the floor;
        # its XML zero pose has the soles 35 mm below it, causing a bounce.
        self.data.joint('root').qpos[2] += .035
        self.data.qpos[self.qpos_ids] = self.trajectory.sample(0.)[0]
        mujoco.mj_forward(self.model, self.data)
        self.ball_start = self.data.body('golf_ball').xpos.copy()
        self.impact_time = None
        self.launch_velocity = None
        self.max_ball_speed = 0.
        self.max_grip_error = 0.
        self.max_tracking_error = 0.
        self.saturated_steps = 0
        self.steps = 0
        self.trace = []

    def step(self):
        # Refresh kinematics and bias BEFORE computing control. The original
        # XML uses Euler, which supports the split mj_step1/mj_step2 interface.
        mujoco.mj_step1(self.model, self.data)
        t = self.data.time
        target, target_speed, target_acceleration = self.trajectory.sample(t)
        mujoco.mj_fullM(self.model, self.data, self.inertia)
        torque = (KP * (target - self.data.qpos[self.qpos_ids])
                  + KD * (target_speed - self.data.qvel[self.qvel_ids])
                  + self.data.qfrc_bias[self.qvel_ids]
                  - self.data.qfrc_passive[self.qvel_ids]
                  + self.inertia[np.ix_(self.qvel_ids, self.qvel_ids)] @ target_acceleration)
        limits = self.model.actuator_ctrlrange[self.motor_ids]
        self.data.ctrl[self.motor_ids] = np.clip(torque, limits[:, 0], limits[:, 1])
        self.saturated_steps += int(np.any((torque < limits[:, 0]) | (torque > limits[:, 1])))
        self.steps += 1

        # External torques (N m), never angle overrides or scripted impulses.
        self.data.qfrc_applied[:] = 0.
        if self.disturbance_joint is not None and SETTLE_TIME <= t <= FINISH_TIME:
            u = (t - SETTLE_TIME) / (FINISH_TIME - SETTLE_TIME)
            disturbance = (self.disturbance_amplitude * np.sin(np.pi*u)**2
                           * np.sin(2*np.pi*self.disturbance_frequency*(t-SETTLE_TIME)))
            dof = self.model.joint(self.disturbance_joint).dofadr[0]
            self.data.qfrc_applied[dof] = disturbance

        # Contacts from step1 belong to time t, before step2 integrates.
        contact = any({c.geom1, c.geom2} == {self.club_id, self.ball_id}
                      for c in self.data.contact)
        if contact and self.impact_time is None:
            self.impact_time = float(t)
        mujoco.mj_step2(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        velocity = self.data.qvel[self.ball_dof:self.ball_dof+3].copy()
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_ball_speed:
            self.max_ball_speed = speed
            if self.impact_time is not None:
                self.launch_velocity = velocity
        grip_error = float(np.linalg.norm(self.data.site('shared_grip').xpos
                                         - self.data.site('left_grip').xpos))
        self.max_grip_error = max(self.max_grip_error, grip_error)
        target, _, _ = self.trajectory.sample(self.data.time)
        error = self.data.qpos[self.qpos_ids] - target
        if self.data.time >= SETTLE_TIME:
            self.max_tracking_error = max(self.max_tracking_error, float(np.max(np.abs(error))))
        self.trace.append(np.r_[self.data.time, self.data.qpos[self.qpos_ids], target,
                                self.data.ctrl[self.motor_ids],
                                self.data.site('club_head').xpos,
                                self.data.body('golf_ball').xpos, velocity, grip_error])

    def run(self, duration=3.5):
        for _ in range(int(np.ceil((duration-self.data.time)/self.model.opt.timestep))):
            self.step()
        return self.metrics()

    def metrics(self):
        ball = self.data.body('golf_ball').xpos
        trace = np.asarray(self.trace)
        return {
            'duration_s': float(self.data.time),
            'impact_time_s': self.impact_time,
            'peak_ball_speed_m_s': self.max_ball_speed,
            'ball_displacement_m': float(np.linalg.norm(ball[:2]-self.ball_start[:2])),
            'ball_final_xyz_m': ball.tolist(),
            'ball_velocity_at_peak_m_s': None if self.launch_velocity is None else self.launch_velocity.tolist(),
            'max_grip_error_mm': self.max_grip_error * 1000.,
            'max_tracking_error_deg': float(np.rad2deg(self.max_tracking_error)),
            'saturated_step_fraction': self.saturated_steps/max(1, self.steps),
            'joint_excursion_deg': (np.rad2deg(np.ptp(trace[:, 1:8], axis=0)).tolist()
                                    if len(trace) else [0.]*7),
            'final_angles_deg': np.rad2deg(self.data.qpos[self.qpos_ids]).tolist(),
            'warnings': {str(mujoco.mjtWarning(i)): int(w.number)
                         for i, w in enumerate(self.data.warning) if w.number},
        }

    def report(self):
        print(json.dumps(self.metrics(), indent=2))

    def save_trace(self, path):
        header = (['time_s'] + [f'{n}_rad' for n in JOINTS]
                  + [f'{n}_target_rad' for n in JOINTS]
                  + [f'{n}_torque_Nm' for n in JOINTS]
                  + ['head_x_m', 'head_y_m', 'head_z_m', 'ball_x_m', 'ball_y_m', 'ball_z_m',
                     'ball_vx_m_s', 'ball_vy_m_s', 'ball_vz_m_s', 'grip_error_m'])
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.trace)


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='run without opening a window')
    parser.add_argument('--duration', type=positive_float, default=3.5, help='simulation seconds')
    parser.add_argument('--slow-motion', type=positive_float, default=1., metavar='FACTOR',
                        help='playback speed divisor, e.g. 3 (physics is unchanged)')
    parser.add_argument('--repeat', action='store_true', help='replay after duration in the viewer')
    parser.add_argument('--csv', type=Path, help='save the measured motion trace')
    parser.add_argument('--disturb-joint', choices=JOINTS, help='apply an external shaky-joint torque')
    parser.add_argument('--disturb-amplitude', type=positive_float, default=3., help='peak disturbance in N m')
    args = parser.parse_args()
    sim = SwingSimulation(disturbance_joint=args.disturb_joint,
                          disturbance_amplitude=args.disturb_amplitude)
    if args.headless:
        sim.run(args.duration)
    else:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera('front_fixed').id
            start = time.monotonic()
            while viewer.is_running():
                frame_start = time.monotonic()
                target_time = min(args.duration, (frame_start-start)/args.slow_motion)
                with viewer.lock():
                    while sim.data.time + 1e-10 < target_time:
                        sim.step()
                viewer.sync()
                if args.repeat and frame_start-start > (args.duration+.8)*args.slow_motion:
                    with viewer.lock():
                        sim.reset()
                    start = time.monotonic()
                time.sleep(max(0., 1./FPS-(time.monotonic()-frame_start)))
    if args.csv:
        sim.save_trace(args.csv)
    sim.report()


if __name__ == '__main__':
    main()
