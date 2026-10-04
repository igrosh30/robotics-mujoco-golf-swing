"""
run_arm.py — drive the MuJoCo 2-link arm from the IK in kinematics.py.

Stage 1: kinematics only. Gravity is off in arm2.xml, so the arm does not
fall or swing. We WRITE the joint angles directly and call mj_forward, which
runs MuJoCo's own forward kinematics (placing every body) without stepping
any dynamics. That makes this a pure geometry check.

PLANE MAPPING — important:
    the hand derivation works in the x-y plane.
    MuJoCo is z-up, and the arm lives in the x-z plane.
        derivation x  ->  MuJoCo x
        derivation y  ->  MuJoCo z
    So an IK target (x, y) becomes the world point (x, 0, y).

Run:
    python run_arm.py                 # headless: numbers only
    mjpython run_arm.py --view        # with the interactive viewer (macOS)

On macOS the passive viewer must be launched with `mjpython`, not `python`.
mjpython ships with the mujoco package and is already on PATH inside the venv.
"""

import argparse
import time

import mujoco
import numpy as np

import kinematics as kin

XML_PATH = "models/arm2.xml"

# Link lengths. These MUST match the geoms in arm2.xml or the verification
# below is meaningless — one source of truth would be better, and making it
# so is one of the exercises.
L1, L2 = 0.50, 0.40


# --------------------------------------------------------------------------

def load():
    """Load the model and allocate its state."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    return model, data


def joint_ids(model):
    """Look joints up by NAME, not by index.

    Indices shift the moment someone reorders the XML; names do not. This is
    the single most important habit for keeping Python and XML in sync.
    """
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "theta1")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "theta2")
    if j1 < 0 or j2 < 0:
        raise RuntimeError("joint 'theta1' or 'theta2' not found in the model")
    return j1, j2


def set_angles(model, data, theta1, theta2):
    """Write the joint angles and recompute all body positions.

    mj_forward runs kinematics + constraint setup but does NOT advance time.
    mj_step would integrate the dynamics; that is Stage 2, not this.
    """
    j1, j2 = joint_ids(model)
    data.qpos[model.jnt_qposadr[j1]] = theta1
    data.qpos[model.jnt_qposadr[j2]] = theta2
    mujoco.mj_forward(model, data)


def tip_world(model, data):
    """World position of the tip site, as MuJoCo computed it."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    return data.site_xpos[sid].copy()      # (x, y, z) in world


def move_target_marker(model, data, x, y):
    """Place the translucent target sphere at the desired point, so the
    viewer shows goal and tip together."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    mid = model.body_mocapid[bid]
    if mid >= 0:
        data.mocap_pos[mid] = np.array([x, 0.0, y])


# --------------------------------------------------------------------------

def verify(model, data, targets, branch=+1):
    """For each target: solve IK, write the angles, ask MuJoCo where the tip
    ended up, and compare.

    This closes the loop across THREE independent implementations of the same
    geometry — the hand derivation, the Python FK, and MuJoCo's internal
    kinematics. If all three agree, the model is right.
    """
    print(f"{'target (x,y)':>18} {'theta1':>10} {'theta2':>10} "
          f"{'mujoco tip (x,z)':>22} {'error':>10}")
    print("-" * 78)

    worst = 0.0
    for (x, y) in targets:
        try:
            t1, t2 = kin.inverse(x, y, L1, L2, branch=branch)
        except ValueError as e:
            print(f"({x:+.2f}, {y:+.2f})  -> {e}")
            continue

        set_angles(model, data, t1, t2)
        tip = tip_world(model, data)
        err = np.hypot(tip[0] - x, tip[2] - y)
        worst = max(worst, err)

        print(f"({x:+.2f}, {y:+.2f})".rjust(18)
              + f"{np.degrees(t1):>10.2f}{np.degrees(t2):>10.2f}"
              + f"   ({tip[0]:+.4f}, {tip[2]:+.4f})".rjust(22)
              + f"{err:>10.2e}")

    print("-" * 78)
    print(f"worst error: {worst:.3e} m")
    return worst


def animate(model, data, xs, ys, branch=+1, hold=0.04):
    """Walk the tip along a path, one IK solve per point."""
    import mujoco.viewer

    t1s, t2s = kin.solve_path(xs, ys, L1, L2, branch=branch)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for t1, t2, x, y in zip(t1s, t2s, xs, ys):
                if not viewer.is_running():
                    break
                if np.isnan(t1):
                    continue
                set_angles(model, data, t1, t2)
                move_target_marker(model, data, x, y)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(hold)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true",
                    help="open the viewer and animate an arc (needs mjpython on macOS)")
    ap.add_argument("--branch", type=int, default=1, choices=(1, -1),
                    help="which elbow configuration to use")
    args = ap.parse_args()

    model, data = load()
    print(f"loaded {XML_PATH}: nq={model.nq} dof(s), "
          f"workspace r in [{abs(L1-L2):.3f}, {L1+L2:.3f}]\n")

    targets = [(0.60, 0.30), (0.20, 0.70), (-0.40, 0.50),
               (0.00, -0.60), (0.85, 0.00), (0.30, 0.30)]
    verify(model, data, targets, branch=args.branch)

    if args.view:
        # A rough swing arc: centred near the shoulder, sweeping downward.
        xs, ys = kin.arc_trajectory(cx=0.0, cy=0.0, radius=0.75,
                                    a_start=np.deg2rad(120),
                                    a_end=np.deg2rad(-60),
                                    n=160)
        print("\nopening viewer — close the window to exit")
        animate(model, data, xs, ys, branch=args.branch)


if __name__ == "__main__":
    main()
