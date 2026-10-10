"""Hand/club path inspired by Figure 1 of the lab guide, with two bending elbows."""
import mujoco
import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SETTLE_TIME = .50
BACKSWING_TIME = 1.00
DOWNSWING_TIME = .75
FOLLOW_TIME = .85
IMPACT_TIME = SETTLE_TIME + BACKSWING_TIME + DOWNSWING_TIME
FINISH_TIME = IMPACT_TIME + FOLLOW_TIME # note simulation runs for 3.5s! can't be higher 
JOINTS = (
    'torso_turn', 'right_arm_lift', 'right_shoulder1', 'right_shoulder2',
    'right_elbow', 'right_wrist_cock', 'right_wrist_deviation',
    'left_arm_lift', 'left_shoulder1', 'left_shoulder2', 'left_elbow',
)


class SwingTrajectory:
    """Solve a smooth reference before simulation; runtime still uses motor torques.

    The shared grip is specified relative to the upper torso. Two-link geometry
    places each elbow below/outside its shoulder-to-hand line. Consequently the
    trail elbow folds on the backswing, and the lead elbow folds on the finish.
    The club direction travels through a tilted swing plane with nonzero speed
    at impact. These are manually designed reference poses, not motion capture.
    """

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qids = np.array([model.joint(n).qposadr[0] for n in JOINTS])
        limits = np.array([model.joint(n).range for n in JOINTS[1:]])
        self.bounds = (limits[:, 0]+1e-5, limits[:, 1]-1e-5)
        # Columns: torso degrees, grip X/Y/Z metres in upper_torso, club phase degrees.
        self.times = np.array([0., SETTLE_TIME, SETTLE_TIME+BACKSWING_TIME, IMPACT_TIME, FINISH_TIME])
        self.progress = BPoly.from_derivatives(self.times,
            [[0.,0.,0.],[0.,0.,0.],[-1.,0.,0.],[0.,2.5,0.],[1.,0.,0.]])
        path_knots = np.array([-1.,-.5,0.,.5,1.])
        self.poses = np.array([
            [-45, .17, -.23, .46, -220],
            [-20, .27, -.10,  .22,  -75],
            [0,   .35, 0,     .08,    0],
            [40,  .27, .13,   .23,  100],
            [65,  .13, .235,  .48,  250],
        ], dtype=float)
        speeds = np.zeros_like(self.poses)
        for i in [1,2,3]:
            speeds[i] = (self.poses[i+1]-self.poses[i-1])/(path_knots[i+1]-path_knots[i-1])
        speeds[2,1:4] = [0.,.26,0.]
        speeds[0] = self.poses[1]-self.poses[0]
        speeds[-1] = self.poses[-1]-self.poses[-2]
        self.path = BPoly.from_derivatives(path_knots,
            [[p,v,np.zeros(5)] for p,v in zip(self.poses,speeds)])
        self.shaft = model.site('club_head').pos.copy()
        self.shaft /= np.linalg.norm(self.shaft)
        self.axis = np.array([.89, 0., .456]); self.axis /= np.linalg.norm(self.axis)
        self.down = np.cross([0.,1.,0.],self.axis)
        self.max_ik_error = 0.
        parameters, poses = [], []
        for direction in [-1,1]:
            seed = np.zeros(len(JOINTS)-1)
            for u in np.linspace(0.,direction,301):
                q, error = self.solve(self.path(u),seed)
                if error > 1e-4:
                    raise ValueError(f'Unreachable swing reference at u={u:.3f}: {error:.6g}')
                self.max_ik_error=max(self.max_ik_error,error)
                if direction == -1 or u != 0:
                    parameters.append(u); poses.append(q)
                seed=q[1:]
        order=np.argsort(parameters)
        self.spline=CubicSpline(np.array(parameters)[order],np.array(poses)[order])

    def solve(self, pose, seed):
        model, data = self.model, self.data
        torso, gx, gy, gz, phase = pose
        grip = np.array([gx,gy,gz])
        data.qpos[self.qids[0]] = np.deg2rad(torso)
        mujoco.mj_kinematics(model,data)
        upper = data.body('upper_torso')
        rotation = upper.xmat.reshape(3,3).copy()
        origin = upper.xpos.copy()
        target = origin + rotation@grip
        elbows = []
        for side, sign in [('right',-1),('left',1)]:
            shoulder = model.body(side+'_upper_arm').pos
            upper_length = np.linalg.norm(model.body(side+'_lower_arm').pos)
            lower_length = np.linalg.norm([.18,.12,-.04])
            delta = grip-shoulder
            distance = np.linalg.norm(delta)
            if not abs(upper_length-lower_length) < distance < upper_length+lower_length:
                raise ValueError(f'{side} hand target lies outside arm reach')
            n = delta/distance
            along = (upper_length**2-lower_length**2+distance**2)/(2*distance)
            # Near impact, rotate the trail forearm to present a slightly
            # upward-facing club face. Away from the ball, keep elbows clear
            # of the chest. The hand/shaft targets remain unchanged.
            pole_x = .35 - (.9*np.exp(-(phase/45.)**2) if side == 'right' else 0.)
            pole = np.array([pole_x,sign*.4,-1.])
            pole -= n*np.dot(n,pole)
            pole /= np.linalg.norm(pole)
            elbow = shoulder+along*n+np.sqrt(max(0.,upper_length**2-along**2))*pole
            elbows.append(origin+rotation@elbow)
        direction = Rotation.from_rotvec(self.axis*np.deg2rad(phase)).apply(self.down)

        def residual(q):
            data.qpos[self.qids[1:]] = q
            mujoco.mj_kinematics(model,data)
            shaft = data.body('right_hand').xmat.reshape(3,3)@self.shaft
            return np.r_[10*(data.site('shared_grip').xpos-target),
                         10*(data.site('left_grip').xpos-target),
                         3*(shaft-direction),
                         10*(data.body('right_lower_arm').xpos-elbows[0]),
                         10*(data.body('left_lower_arm').xpos-elbows[1]),
                         .0002*(q-seed)]

        result = least_squares(residual,np.clip(seed,*self.bounds),bounds=self.bounds,
            max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        error = np.linalg.norm(residual(result.x)[:15])
        return np.r_[np.deg2rad(torso),result.x], error

    def sample(self,t):
        t = float(np.clip(t,0,FINISH_TIME))
        u,du,ddu = self.progress(t), self.progress(t,1), self.progress(t,2)
        return (self.spline(u), self.spline(u,1)*du,
                self.spline(u,2)*du**2+self.spline(u,1)*ddu)
