"""Export only front/side GIFs and front/side picture sheets."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from main import (SwingSimulation, SETTLE_TIME, BACKSWING_TIME,
                      IMPACT_TIME, FINISH_TIME, positive_float)

ROOT = Path(__file__).resolve().parent


def phase(t):
    t += 1e-9
    if t < SETTLE_TIME:
        return 'Address'
    if t < SETTLE_TIME + BACKSWING_TIME:
        return 'Backswing'
    if t < IMPACT_TIME:
        return 'Downswing'
    if t < FINISH_TIME:
        return 'Follow-through'
    return 'Hold'


def export_visuals(output, duration=3.5, slow_motion=2.):
    """Run the unchanged physics, then render saved poses on separate data.

    Only four files are written. Re-running replaces these same four files.
    Each PNG contains the main poses, so there are no loose snapshot files.
    """
    sim = SwingSimulation()
    states, times = [sim.data.qpos.copy()], [0.]
    print('Simulating swing...', flush=True)
    for _ in range(int(np.ceil(duration / sim.model.opt.timestep))):
        sim.step()
        states.append(sim.data.qpos.copy())
        times.append(sim.data.time)
    times = np.asarray(times)
    data = mujoco.MjData(sim.model)
    frame_times = np.arange(0, times[-1] + 1e-9, 1 / 30)
    indices = np.unique(np.minimum(np.searchsorted(times, frame_times), len(times)-1))
    def progress_time(value, start, end):
        ascending = sim.trajectory.progress(end) > sim.trajectory.progress(start)
        for _ in range(45):
            mid = (start+end)/2
            if (sim.trajectory.progress(mid) < value) == ascending:
                start = mid
            else:
                end = mid
        return (start+end)/2

    top = SETTLE_TIME + BACKSWING_TIME
    events = [
        ('Address', 0.),
        ('Takeaway', progress_time(-.25,SETTLE_TIME,top)),
        ('Backswing', progress_time(-.65,SETTLE_TIME,top)),
        ('Backswing top',top),
        ('Early downswing',progress_time(-.65,top,IMPACT_TIME)),
        ('Late downswing',progress_time(-.25,top,IMPACT_TIME)),
        ('First contact' if sim.impact_time is not None else 'Planned strike (no contact)',
         sim.impact_time if sim.impact_time is not None else IMPACT_TIME),
        ('Release',progress_time(.25,IMPACT_TIME,FINISH_TIME)),
        ('Follow-through',progress_time(.6,IMPACT_TIME,FINISH_TIME)),
        ('Finish',FINISH_TIME),
    ]
    events = [(label, t) for label, t in events if t <= times[-1] + 1e-9]
    output.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(sim.model, height=600, width=800) as renderer:
        for name, camera in [('front', 'front_fixed'), ('side', 'side')]:
            print(f'Rendering {name} view...', flush=True)

            def frame(index, label=None, animation=False):
                data.qpos[:] = states[index]
                data.time = float(times[index])
                mujoco.mj_forward(sim.model, data)
                renderer.update_scene(data, camera=camera)
                im = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(im)
                draw.rectangle((6, 6, 794, 44), fill=(15, 29, 39))
                draw.text((14, 11), f'{name.title()} view | {label or phase(data.time)} | t = {data.time:.3f} s', fill='white')
                if animation:
                    draw.text((14, 27), f'Anchored lower body | Playback {slow_motion:g}x slower', fill='white')
                else:
                    draw.text((14, 27), 'Anchored lower body', fill='white')
                return im

            frames = [frame(int(i), animation=True).convert('P', palette=Image.Palette.ADAPTIVE)
                      for i in indices]
            frames[0].save(output / f'{name}.gif', save_all=True,
                           append_images=frames[1:], duration=max(10, round(1000*slow_motion/30)), loop=0)
            sheet = Image.new('RGB', (3000, 450*((len(events)+4)//5)), 'white')
            for i, (label, t) in enumerate(events):
                index = min(int(np.searchsorted(times, t-1e-10)), len(times)-1)
                sheet.paste(frame(index, label).resize((600,450),Image.Resampling.LANCZOS),
                            ((i % 5)*600, (i//5)*450))
            sheet.save(output / f'{name}.png')
    print(f'Done: {output}\nfront.gif, side.gif, front.png, side.png', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports',
                        help='destination; the same four visuals are replaced each run')
    parser.add_argument('--duration', type=positive_float, default=3.5, help='simulation seconds')
    parser.add_argument('--slow-motion', type=positive_float, default=2., help='GIF playback divisor')
    args = parser.parse_args()
    export_visuals(args.output.resolve(), args.duration, args.slow_motion)


if __name__ == '__main__':
    main()
