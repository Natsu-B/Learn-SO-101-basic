"""Record actual control-step states at 30 fps, with episode JSON lines on stdout.

Run on the main thread (CGL on macOS, hidden GLFW elsewhere). Requires ffmpeg and
the learning extra (SAC, Pillow). No physics steps are added for rendering/holds.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_sim import CONTROL_PERIOD, PickPlaceRenderer, PickPlaceSimulator
from robot_sim.pick_place_pipeline import run_pick_place_episode, summarize_pick_place

FPS = 30
DEFAULT_MODEL = Path("runs/so101-transport/seed-42/best.zip")


class VideoWriter:
    """Stream RGB frames; publish the MP4 only after a successful encoder exit."""

    def __init__(self, output, width, height):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is missing; enter nix develop with ffmpeg installed")
        self.output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", prefix=".record-",
                                             dir=output.parent, delete=False)
        self.temporary = Path(handle.name)
        handle.close()
        self.frames = 0
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.temporary),
            ], stdin=subprocess.PIPE, stderr=self.errors)
        except BaseException:
            self.temporary.unlink(missing_ok=True)
            self.errors.close()
            raise

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self, *, publish):
        try:
            # Broken pipes still need wait(), stderr collection, and cleanup.
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            if publish and code:
                self.errors.seek(0)
                raise RuntimeError(f"ffmpeg exited {code}: {self.errors.read().decode(errors='replace')}")
            if publish:
                self.temporary.replace(self.output)
        finally:
            self.errors.close()
            self.temporary.unlink(missing_ok=True)


class Overlay:
    def __init__(self, width, height, transport):
        self.width, self.height = width, height
        self.transport = transport
        self.scale = width / 1280
        self.font = ImageFont.load_default(size=max(12, round(25 * self.scale)))
        self.small = ImageFont.load_default(size=max(10, round(20 * self.scale)))

    def frame(self, rgb, state, *, seed, episode, total, skill, learned, result=None):
        image = Image.frombytes("RGB", (self.width, self.height), rgb)
        draw = ImageDraw.Draw(image)
        pad = max(8, round(20 * self.scale))
        line = max(17, round(35 * self.scale))
        header = pad * 2 + 3 * line
        draw.rectangle((0, 0, self.width, header), fill=(17, 24, 36))
        draw.text((pad, pad), f"SO-101  |  Episode {episode}/{total}  |  Seed {seed}",
                  font=self.font, fill="white")
        color = (72, 224, 170) if learned else (123, 189, 255)
        label = "LEARNED (SAC)" if learned else "IK (scripted)"
        draw.text((pad, pad + line), f"{skill.upper()}  |  {label}", font=self.font, fill=color)
        error_mm = np.linalg.norm(np.asarray(state["cube_pos"])[:2] - np.asarray(state["target"])[:2]) * 1000
        draw.text((pad, pad + 2 * line), f"Cube -> target XY error: {error_mm:.2f} mm",
                  font=self.small, fill=(224, 230, 238))
        footer = self.height - pad - 2 * line
        draw.rectangle((0, footer - pad, self.width, self.height), fill=(17, 24, 36))
        if result is None:
            draw.text((pad, footer), "Grasp: IK   >   Transport: SAC   >   Place: IK" if self.transport == "learned"
                      else "Grasp: IK   >   Transport: IK   >   Place: IK",
                      font=self.small, fill=(224, 230, 238))
            draw.text((pad, footer + line), "Red cube  |  Green target  |  30 fps",
                      font=self.small, fill=(224, 230, 238))
        else:
            status = "SUCCESS" if result["success"] else "FAILURE"
            color = (72, 224, 170) if result["success"] else (255, 110, 110)
            draw.text((pad, footer), f"{status}  |  Final XY: {result['final_xy_error'] * 1000:.2f} mm",
                      font=self.font, fill=color)
            draw.text((pad, footer + line), "Rust predicates  |  Hold: physics paused",
                      font=self.small, fill="white")
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 25, 42, 99])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=Path("runs/video/pick_place_demo.mp4"))
    parser.add_argument("--transport", choices=("learned", "scripted"), default="learned")
    parser.add_argument("--hold-frames", type=int, default=90)
    args = parser.parse_args()
    if any(not 0 <= seed < 2**64 for seed in args.seeds):
        parser.error("seeds must fit in uint64")
    if (not 320 <= args.width <= 3840 or not 240 <= args.height <= 2160
            or args.width % 2 or args.height % 2):
        parser.error("use even dimensions within 320..3840 x 240..2160")
    if args.hold_frames < 0:
        parser.error("--hold-frames must be non-negative")
    if not np.isclose(CONTROL_PERIOD, 1 / FPS):
        parser.error("control period must be 1/30 s for real-time recording")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is missing; enter nix develop with ffmpeg installed")
    # Keep recording outputs separate from protected training artifacts.
    protected = Path("runs/so101-transport").resolve()
    if args.output.resolve().is_relative_to(protected):
        parser.error("--output must be outside runs/so101-transport/")

    model = load_env = sim = renderer = writer = None
    complete = False
    try:
        options = {}
        if args.transport == "learned":
            import torch
            from stable_baselines3 import SAC
            from robot_sim.transport_env import GoalTransportEnv

            config = json.loads((args.model.parent / "config.json").read_text())
            if config.get("environment_version") != "goal-transport-v1":
                parser.error("--model must be a GoalTransportEnv checkpoint")
            torch.set_num_threads(1)
            load_env = GoalTransportEnv(**config["env"])
            sim = load_env.sim
            model = SAC.load(args.model, env=load_env, device="cpu")
            options = dict(max_transport_steps=load_env.horizon,
                           max_delta=load_env.max_delta, rate_limit=load_env.rate_limit)
        else:
            sim = PickPlaceSimulator()
        renderer = PickPlaceRenderer(sim, args.width, args.height)
        writer = VideoWriter(args.output, args.width, args.height)
        overlay = Overlay(args.width, args.height, args.transport)
        outcomes = []
        stills = {}
        # Capture a useful late frame from the first episode of each skill.
        for episode, seed in enumerate(args.seeds, start=1):
            moments = {}
            last = None

            def on_step(state, skill, learned):
                nonlocal last
                rgb = renderer.render()
                image = overlay.frame(rgb, state, seed=seed, episode=episode,
                                      total=len(args.seeds), skill=skill, learned=learned)
                writer.write(image)
                last = (rgb, state, skill, learned)
                if episode == 1:
                    moments[skill] = (image, writer.frames - 1)

            row = run_pick_place_episode(seed, sim=sim, model=model, transport=args.transport,
                                         on_step=on_step, **options)
            row["type"] = "episode"
            if last is not None:
                rgb, state, skill, learned = last
                image = overlay.frame(rgb, state, seed=seed, episode=episode,
                                      total=len(args.seeds), skill=skill, learned=learned, result=row)
                for _ in range(args.hold_frames):
                    writer.write(image)
            for skill, (image, frame_index) in moments.items():
                path = args.output.with_name(f"{args.output.stem}_{skill}.png")
                image.save(path)
                stills[skill] = dict(path=str(path), frame=frame_index,
                                     time_seconds=frame_index / FPS)
            row["video_frames"] = row["control_steps"] + args.hold_frames
            outcomes.append(row)
            print(json.dumps(row), flush=True)
            print(f"Seed {seed}: {'SUCCESS' if row['success'] else 'FAILURE'}, "
                  f"XY {row['final_xy_error'] * 1000:.6f} mm, {row['control_steps']} steps",
                  file=sys.stderr, flush=True)
        summary = summarize_pick_place(outcomes, transport=args.transport)
        summary.update(seeds=args.seeds, output=str(args.output), fps=FPS,
                       width=args.width, height=args.height, frames=writer.frames,
                       duration_seconds=writer.frames / FPS, hold_frames=args.hold_frames, stills=stills)
        if model is not None:
            summary.update(model=str(args.model), training_steps=model.num_timesteps)
        complete = True
    finally:
        try:
            if writer is not None:
                writer.close(publish=complete)
        finally:
            if renderer is not None:
                renderer.close()
            if load_env is not None:
                load_env.close()
            elif sim is not None:
                sim.close()
    summary["file_size_bytes"] = args.output.stat().st_size
    # Sidecars make the exact seeds, results, timing and extracted moments reviewable.
    args.output.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in outcomes))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
