import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from main import SwingSimulation

ROOT = Path(__file__).resolve().parent


def run_case(case, duration):
    sim = SwingSimulation(settling_time=case["settling_time"])

    states = [sim.data.qpos.copy()]
    times = [0.0]
    while sim.data.time < duration - 1e-12:
        sim.step()
        states.append(sim.data.qpos.copy())
        times.append(sim.data.time)
    return sim, np.asarray(states), np.asarray(times)


def render_gif(sim, states, times, output, slow_motion):
    """Render only the front camera, keeping one GIF in each case folder."""
    output.mkdir(parents=True, exist_ok=True)
    frame_times = np.arange(0.0, times[-1] + 1e-12, 1.0 / 30.0)
    indices = np.unique(np.minimum(np.searchsorted(times, frame_times), len(times) - 1))
    data = mujoco.MjData(sim.model)

    with mujoco.Renderer(sim.model, height=600, width=800) as renderer:
        frames = []
        for index in indices:
            data.qpos[:] = states[index]
            data.time = float(times[index])
            mujoco.mj_forward(sim.model, data)
            renderer.update_scene(data, camera="front_fixed")
            image = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(image)
            draw.rectangle((6, 6, 794, 42), fill=(15, 29, 39))
            draw.text(
                (14, 11),
                f"Front view | t = {data.time:.3f} s",
                fill="white",
            )
            frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))

        frames[0].save(
            output / "front.gif",
            save_all=True,
            append_images=frames[1:],
            duration=max(10, round(1000 * slow_motion / 30)),
            loop=0,
        )


def case_folder_name(case):
    name = case["name"].replace(".", "p")
    return name.replace(" ", "_")


def run_cases(cases, output_root, duration, slow_motion):
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in cases:
        folder = output_root / case_folder_name(case)
        print(f"Running {case['name']}...", flush=True)
        sim, states, times = run_case(case, duration)
        render_gif(sim, states, times, folder, slow_motion)
        metrics = {
            "case": case,
            "duration_requested_s": duration,
            "gif": "front.gif",
            **sim.metrics(),
        }
        (folder / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        summary.append(metrics)
        print(f"  -> {folder / 'front.gif'}", flush=True)

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def all_cases():
    return [
        {"name": f"pd_settling_{value:g}s", "settling_time": value}
        for value in (0.3, 0.75, 5, 10, 15, 20)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the complete sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports" / "analysis",
        help="root folder for experiment outputs",
    )
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--slow-motion", type=float, default=2.0)
    args = parser.parse_args()

    if not args.all:
        parser.error("use --all to run the experiment sweep")
    if args.duration <= 0 or args.slow_motion <= 0:
        parser.error("--duration and --slow-motion must be positive")
    run_cases(
        all_cases(),
        args.output.resolve(),
        args.duration,
        args.slow_motion,
    )


if __name__ == "__main__":
    main()
