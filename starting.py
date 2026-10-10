import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


from swing_trajectory import (SwingTrajectory, JOINTS, SETTLE_TIME, BACKSWING_TIME,
                              DOWNSWING_TIME, FOLLOW_TIME, IMPACT_TIME, FINISH_TIME)

FPS = 60
#gains for each joint actuator 
KP = np.array([650.,300.,220.,220.,180.,100.,100.,300.,220.,220.,180.])
KD = np.array([70.,40.,28.,28.,20.,12.,12.,40.,28.,28.,20.])


class SwingSimulation:
    def __init__(self, *, timestep=None, disturbance_joint=None, disturbance_amplitude=3., disturbance_frequency=8.):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('starting.xml')))
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.trajectory = SwingTrajectory(self.model) #cals swing_trajectory -> returns the initial pose
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.motor_ids = np.array([self.model.actuator(f'{n}_motor').id for n in JOINTS])
        self.club_id = self.model.geom('club_head_collision').id
        self.ball_id = self.model.geom('golf_ball_geom').id
        self.ball_dof = self.model.joint('golf_ball_free').dofadr[0]
        self.inertia = np.empty((self.model.nv, self.model.nv))
        if disturbance_joint is not None and disturbance_joint not in JOINTS:
            raise ValueError(f'Unknown upper-body joint: {disturbance_joint}')
        self.disturbance_joint = disturbance_joint
        self.disturbance_amplitude = disturbance_amplitude
        self.disturbance_frequency = disturbance_frequency
        self.pd_disabled = False #PD enabling
        self.reset()
        

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # The XML anchors the lower body with the soles at ground level.
        self.data.qpos[self.qpos_ids] = self.trajectory.sample(0.)[0]
        mujoco.mj_forward(self.model, self.data)
        self.ball_start = self.data.body('golf_ball').xpos.copy()
        self.impact_time = None
        self.launch_velocity = None
        self.separation_velocity = None
        self.club_speed_at_impact = None
        self.head_was_touching_ball = False
        self.max_ball_height = float(self.data.body('golf_ball').xpos[2])
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
        # Implicit PD: elbow DOFs introduce low-inertia directions in which
        # explicit high-gain damping can oscillate even while holding a pose.
        mass = self.inertia[np.ix_(self.qvel_ids, self.qvel_ids)]
        dt = self.model.opt.timestep
        acceleration = np.linalg.solve(mass + np.diag(dt*KD + dt*dt*KP),
            KP*(target-self.data.qpos[self.qpos_ids])
            + (KD+dt*KP)*(target_speed-self.data.qvel[self.qvel_ids])
            + mass@target_acceleration)
        torque = (mass@acceleration + self.data.qfrc_bias[self.qvel_ids]
                  - self.data.qfrc_passive[self.qvel_ids])
        limits = self.model.actuator_ctrlrange[self.motor_ids]

        pd_constant = 0
        if self.pd_disabled:
            pd_constant = 0.25
        else:
            pd_constant = 1
        
        self.data.ctrl[self.motor_ids] = np.clip(torque, limits[:, 0], limits[:, 1]) * pd_constant
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
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jacSite(self.model,self.data,jac,None,self.model.site('club_head').id)
            self.club_speed_at_impact = float(np.linalg.norm(jac@self.data.qvel))
        if self.head_was_touching_ball and not contact and self.separation_velocity is None:
            self.separation_velocity = self.data.qvel[self.ball_dof:self.ball_dof+3].copy()
        self.head_was_touching_ball = contact
        mujoco.mj_step2(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        velocity = self.data.qvel[self.ball_dof:self.ball_dof+3].copy()
        speed = float(np.linalg.norm(velocity))
        self.max_ball_height = max(self.max_ball_height,float(self.data.body('golf_ball').xpos[2]))
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
            'club_head_speed_at_impact_m_s': self.club_speed_at_impact,
            'ball_velocity_after_impact_m_s': None if self.separation_velocity is None else self.separation_velocity.tolist(),
            'max_ball_height_m': self.max_ball_height,
            'peak_ball_speed_m_s': self.max_ball_speed,
            'ball_displacement_m': float(np.linalg.norm(ball[:2]-self.ball_start[:2])),
            'ball_final_xyz_m': ball.tolist(),
            'ball_velocity_at_peak_m_s': None if self.launch_velocity is None else self.launch_velocity.tolist(),
            'max_grip_error_mm': self.max_grip_error * 1000.,
            'max_tracking_error_deg': float(np.rad2deg(self.max_tracking_error)),
            'saturated_step_fraction': self.saturated_steps/max(1, self.steps),
            'joint_excursion_deg': (np.rad2deg(np.ptp(trace[:, 1:1+len(JOINTS)], axis=0)).tolist()
                                    if len(trace) else [0.]*len(JOINTS)),
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
    start = 0
    reset_flag = [False]
    pd_disabled = [False]

    #click R - reset 
    def on_key(key_code):
        if key_code == 82:
            reset_flag[0] = True
            #print("reset!")
        if key_code == 80:
            pd_disabled[0] = True

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
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera('front_fixed').id
            start = time.monotonic()
            while viewer.is_running():
                if reset_flag[0]:
                    with viewer.lock():
                        sim.reset()
                    start = time.monotonic()
                    reset_flag[0] = False
                if pd_disabled[0]:
                    sim.pd_disabled = not sim.pd_disabled
                    pd_disabled[0] = False
                
                frame_start = time.monotonic()
                target_time = min(args.duration, (frame_start - start) / args.slow_motion)
                with viewer.lock():
                    while sim.data.time + 1e-10 < target_time:
                        sim.step()
                viewer.sync()
                if args.repeat and frame_start - start > (args.duration + .8) * args.slow_motion:
                    with viewer.lock():
                        sim.reset()
                    start = time.monotonic()
                time.sleep(max(0., 1. / FPS - (time.monotonic() - frame_start)))
    if args.csv:
        sim.save_trace(args.csv)
    sim.report()


if __name__ == '__main__':
    main()
