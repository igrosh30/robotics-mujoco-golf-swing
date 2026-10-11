import json
import argparse
from pathlib import Path

from main import SwingSimulation, JOINTS


AMPLITUDE = 3.0   # N·m, same for all joints
DURATION = 3.5     # seconds

#mjpython disturbance_sweep.py - run this 

def run_one(joint_name=None, amplitude = AMPLITUDE):
    sim = SwingSimulation(disturbance_joint=joint_name,
                          disturbance_amplitude=amplitude)
    sim.run(DURATION)
    return sim.metrics()


def main():

    parser = argparse.ArgumentParser(description="Disturbance sweep across all joints")
    parser.add_argument('--amplitude', type=float, default= AMPLITUDE, help='disturbance amplitude in N·m')
    args = parser.parse_args()
    amplitude = args.amplitude

    # baseline
    print("Running baseline (no disturbance)...")
    baseline = run_one(None,0)

    # disturbed cases
    print(f"Running with {amplitude} Nm of disturbance...")
    results = {}
    for name in JOINTS:
        print(f"Running disturbance on {name}...")
        results[name] = run_one(name, amplitude)

    # print comparison table
    print()
    print(f"{'joint':<28} {'club speed (m/s)':>16} {'ball dist (m)':>14} "
          f"{'track err (deg)':>16} {'saturated (%)':>14}")
    print("-" * 92)

    print(f"{'BASELINE':<28} {baseline['club_head_speed_at_impact_m_s']:>16.3f} "
          f"{baseline['ball_displacement_m']:>14.3f} "
          f"{baseline['max_tracking_error_deg']:>16.3f} "
          f"{baseline['saturated_step_fraction']*100:>14.2f}")
    print("-" * 92)

    for name in JOINTS:
        m = results[name]
        speed = m['club_head_speed_at_impact_m_s'] or 0.0
        dist = m['ball_displacement_m']
        track = m['max_tracking_error_deg']
        sat = m['saturated_step_fraction'] * 100
        print(f"{name:<28} {speed:>16.3f} {dist:>14.3f} {track:>16.3f} {sat:>14.2f}")

    # save raw data
    output = Path(f"disturbance_results_{amplitude:g}Nm.json")
    all_data = {"amplitude_Nm": amplitude, "baseline": baseline, "disturbed": results}
    output.write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    print(f"\nFull metrics saved to {output}")


if __name__ == "__main__":
    main()