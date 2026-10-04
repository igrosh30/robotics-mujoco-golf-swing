"""
kinematics.py- computes forward and inverse kinematics of a 2-link planar arm.

Deliberately has NO MuJoCo import. This is pure geometry, so it can be
tested, plotted and reasoned about on its own. The simulator is a separate
module that consumes this one.

Convention (matches the hand derivation):
    - origin at the shoulder
    - theta1 measured from the +x axis, counter-clockwise positive
    - theta2 measured RELATIVE to link 1, counter-clockwise positive
    - all angles in RADIANS
"""

import numpy as np


# --------------------------------------------------------------------------
# FORWARD KINEMATICS:  angles -> tip position
# --------------------------------------------------------------------------

def forward(theta1, theta2, L1, L2):
    """Return (x, y) of the tip for the given joint angles.

    This is the chained-transform result:
        x = L1*cos(t1) + L2*cos(t1 + t2)
        y = L1*sin(t1) + L2*sin(t1 + t2)
    The (t1 + t2) comes from R(t1) @ R(t2) = R(t1 + t2).
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y


def elbow_position(theta1, L1):
    """Return (x, y) of the elbow. Depends on theta1 ONLY — theta2 is
    downstream and cannot move the joint it hangs off."""
    return L1 * np.cos(theta1), L1 * np.sin(theta1)


# --------------------------------------------------------------------------
# REACHABILITY
# --------------------------------------------------------------------------

def is_reachable(x, y, L1, L2, tol=1e-9):
    """True if the tip can be placed at (x, y).

    Two bounds:
      outer: r <= L1 + L2        (cannot stretch past full extension)
      inner: r >= |L1 - L2|      (cannot fold tighter than fully folded)
    The inner bound is the hole in the middle of the workspace; it vanishes
    when L1 == L2.
    """
    r = np.hypot(x, y)
    return (r <= L1 + L2 + tol) and (r >= abs(L1 - L2) - tol)


# --------------------------------------------------------------------------
# INVERSE KINEMATICS:  tip position -> angles
# --------------------------------------------------------------------------

def inverse(x, y, L1, L2, branch=+1):
    """Return (theta1, theta2) placing the tip at (x, y).

    branch = +1  -> one elbow configuration
    branch = -1  -> the mirrored one
    Both are geometrically valid; the choice is a modelling decision, not
    something the equations can settle.

    Raises ValueError if the target is outside the workspace.
    """
    if branch not in (+1, -1):
        raise ValueError("branch must be +1 or -1")

    r2 = x * x + y * y
    r = np.sqrt(r2)

    if not is_reachable(x, y, L1, L2):
        raise ValueError(
            f"target ({x:.3f}, {y:.3f}) is unreachable: "
            f"r={r:.3f}, allowed [{abs(L1 - L2):.3f}, {L1 + L2:.3f}]"
        )

    # ---- theta2 from the law of cosines on the shoulder-elbow-target triangle
    #      cos(theta2) = (r^2 - L1^2 - L2^2) / (2*L1*L2)
    #      clip guards against floating-point values like 1.0000000002, which
    #      would make arccos return NaN at the exact workspace boundary.
    cos_t2 = (r2 - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)
    theta2 = branch * np.arccos(np.clip(cos_t2, -1.0, 1.0))

    # ---- theta1 = (direction to target) - (offset of link 1 from that line)
    #      atan2 is used instead of arctan(y/x) so every quadrant is handled.
    #      The second atan2 encodes the law-of-cosines angle alpha, and it
    #      carries the sign of theta2 automatically — so the same line serves
    #      both branches with no extra case analysis.
    beta = np.arctan2(y, x)
    alpha = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = beta - alpha

    return theta1, theta2


# --------------------------------------------------------------------------
# SELF-CHECK:  the round trip
# --------------------------------------------------------------------------

def roundtrip_error(x, y, L1, L2, branch=+1):
    """Run IK then FK and report how far the tip lands from the target.

    This is the single most useful test in the whole module. If IK is
    correct, feeding its output back through FK must return the original
    point, to machine precision. Any real error shows up here immediately.
    """
    t1, t2 = inverse(x, y, L1, L2, branch=branch)
    xr, yr = forward(t1, t2, L1, L2)
    return np.hypot(xr - x, yr - y)


# --------------------------------------------------------------------------
# TRAJECTORY HELPERS (used later to describe a swing path)
# --------------------------------------------------------------------------

def arc_trajectory(cx, cy, radius, a_start, a_end, n):
    """n points along a circular arc centred at (cx, cy).

    Returns two arrays (xs, ys). A swing is approximately an arc, so this is
    a reasonable first path to drive the tip along.
    """
    angles = np.linspace(a_start, a_end, n)
    return cx + radius * np.cos(angles), cy + radius * np.sin(angles)


def solve_path(xs, ys, L1, L2, branch=+1):
    """Run IK over a whole path. Returns arrays of theta1 and theta2.

    Points outside the workspace are reported rather than silently skipped,
    because a trajectory that leaves the workspace is a design error worth
    seeing, not something to paper over.
    """
    t1s, t2s, bad = [], [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            a, b = inverse(x, y, L1, L2, branch=branch)
            t1s.append(a)
            t2s.append(b)
        except ValueError:
            bad.append(i)
            t1s.append(np.nan)
            t2s.append(np.nan)
    if bad:
        print(f"[solve_path] {len(bad)} of {len(xs)} points unreachable "
              f"(indices {bad[:10]}{'...' if len(bad) > 10 else ''})")
    return np.array(t1s), np.array(t2s)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    L1, L2 = 0.50, 0.40

    print(f"L1 = {L1}, L2 = {L2}")
    print(f"workspace: inner r = {abs(L1 - L2):.3f}, outer r = {L1 + L2:.3f}\n")

    targets = [(0.6, 0.3), (0.2, 0.7), (-0.4, 0.5), (0.0, -0.6), (0.85, 0.0)]

    for (x, y) in targets:
        for branch in (+1, -1):
            try:
                t1, t2 = inverse(x, y, L1, L2, branch=branch)
                err = roundtrip_error(x, y, L1, L2, branch=branch)
                print(f"target ({x:+.2f}, {y:+.2f})  branch {branch:+d}  "
                      f"theta1 = {np.degrees(t1):+8.3f} deg   "
                      f"theta2 = {np.degrees(t2):+8.3f} deg   "
                      f"roundtrip error = {err:.2e}")
            except ValueError as e:
                print(f"target ({x:+.2f}, {y:+.2f})  branch {branch:+d}  -> {e}")
        print()
