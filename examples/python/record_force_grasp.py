"""Record grasp-force episodes for the demo video (docs/FORCE_CONTROL.md §11).

Each seed is played once per controller with the same object, goal force and
sensor noise, at the real control rate (30 fps, one frame per control period).
The camera looks across the jaw closing axis. The overlay shows the goal
force, the measured jaw forces, the arm correction and the face compression;
object stiffness/mass/friction are shown as ground truth the policy does not
observe. Run on the main thread inside `nix develop` (needs ffmpeg).

  python examples/python/record_force_grasp.py --output runs/video/force_grasp_demo.mp4 \
      --seeds 8000003 8000011 --controllers pi \
      ppo:P10b=runs/force-grasp-coop/p10b-arm-gripper-resume-to-1m/best.zip
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from record_pick_place import FPS, VideoWriter
from robot_sim import CONTROL_PERIOD, ForceGraspRenderer, ForcePi
from robot_sim.force_grasp_env import (ArmAssistPi, ForceGraspConfig, ForceGraspEnv,
                                       pi_controller)

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_LIGHT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
LABELS = {
    "pi": ("力PI（グリッパのみ）", "論文のベースラインの構造。片側のジョーで物体を押し寄せる"),
    "armpi": ("腕補正PI（スクリプト）", "片側が触れたら腕を横に動かして固定ジョーを寄せる"),
    "ppo": ("PPO：腕＋グリッパ（学習済み方策の推論）", "グリッパの指令と腕の横移動を同時に出す"),
}


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # non-macOS: fall back to Pillow's font (Latin only)
        return ImageFont.load_default(size=size)


class Overlay:
    def __init__(self, width, height):
        self.w, self.h = width, height
        s = width / 1280
        self.big = font(FONT, round(28 * s))
        self.mid = font(FONT, round(22 * s))
        self.small = font(FONT_LIGHT, round(19 * s))
        self.pad, self.line = round(18 * s), round(34 * s)

    def frame(self, rgb, env, *, kind, name, seed, step, result=None):
        image = Image.frombytes("RGB", (self.w, self.h), rgb)
        d = ImageDraw.Draw(image)
        pad, line = self.pad, self.line
        d.rectangle((0, 0, self.w, pad * 2 + 3 * line), fill=(17, 24, 36))
        title, note = LABELS[kind]
        color = (72, 224, 170) if kind == "ppo" else (123, 189, 255)
        d.text((pad, pad), f"把持力制御  |  seed {seed}  |  {title}", font=self.big, fill=color)
        d.text((pad, pad + line), note, font=self.small, fill=(224, 230, 238))
        o = env.object
        d.text((pad, pad + 2 * line),
               f"物体の真値（方策には見せない）: 剛性 {o['stiffness']:.0f} N/m,"
               f" 質量 {o['mass'] * 1000:.0f} g, 摩擦 {o['friction']:.2f}, 最大圧縮 {o['travel'] * 1000:.1f} mm",
               font=self.small, fill=(255, 214, 120))
        r = env._reading
        f = r["jaw_force"]
        comp = max(r["deformation"]) / o["travel"]
        top = self.h - pad * 2 - 2 * line
        d.rectangle((0, top, self.w, self.h), fill=(17, 24, 36))
        if result is None:
            d.text((pad, top + pad),
                   f"目標力 {env.goal:.2f} N   固定/可動ジョーの力 {f[0]:.2f} / {f[1]:.2f} N   "
                   f"腕の横移動 {r['pinch_offset'] * 1000:+.1f} mm   圧縮 {100 * min(comp, 1):.0f} %",
                   font=self.mid, fill="white")
            d.text((pad, top + pad + line),
                   f"t = {step * CONTROL_PERIOD:.2f} s   青：ばねで支えた柔らかい面   黒：剛体の芯   "
                   f"30 fps（実時間）", font=self.small, fill=(224, 230, 238))
        else:
            if result["success"]:
                status, scol = "成功：潰さずに目標力で把持", (72, 224, 170)
            elif result["crushed"]:
                status, scol = "失敗：物体を潰した（最大圧縮に到達）", (255, 110, 110)
            elif result["invalid_contact"]:
                status, scol = "失敗：無効な接触（数値的なめり込み）", (255, 170, 90)
            else:
                status, scol = "失敗：目標力に届かないか、接触を保てない", (255, 170, 90)
            d.text((pad, top + pad), status, font=self.big, fill=scol)
            d.text((pad, top + pad + line),
                   f"最大法線力 {result['peak_force_physics']:.1f} N（5 msごとに計測）   "
                   f"最後の1 sの力の誤差 {result['mean_abs_error']:.2f} N   物体の移動 {result['object_displacement_mm']:.1f} mm",
                   font=self.small, fill="white")
        return image


def make_controller(spec):
    if spec == "pi":
        return "pi", "pi", None, ForceGraspConfig()
    if spec == "armpi":
        return "armpi", "armpi", None, ForceGraspConfig()
    if spec.startswith("ppo:"):
        from stable_baselines3 import PPO

        name, path = spec[4:].split("=", 1)
        cfg = ForceGraspConfig.from_dict(json.loads((Path(path).parent / "config.json").read_text())["env"])
        return "ppo", name, PPO.load(path, device="cpu"), cfg
    raise SystemExit(f"unknown controller {spec}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--controllers", nargs="+", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hold-frames", type=int, default=60)
    p.add_argument("--azimuth-offset", type=float, default=90.0)
    a = p.parse_args()
    if not shutil.which("ffmpeg"):
        p.error("ffmpeg is missing; enter nix develop")
    if a.output.exists():
        p.error(f"{a.output} exists; refusing to overwrite")
    controllers = [make_controller(s) for s in a.controllers]
    writer = VideoWriter(a.output, a.width, a.height)
    overlay = Overlay(a.width, a.height)
    rows, complete = [], False
    envs = {}
    try:
        for seed in a.seeds:
            for kind, name, model, cfg in controllers:
                key = json.dumps(cfg.to_dict(), sort_keys=True)
                env = envs.get(key) or envs.setdefault(key, ForceGraspEnv(cfg))
                env.set_progress(1.0)
                obs, _ = env.reset(options={"episode_seed": seed})
                renderer = ForceGraspRenderer(env.sim, a.width, a.height)
                z = env.sim.closing_direction()
                pos = env._reading["object_position"]
                renderer.set_camera([pos[0], pos[1], pos[2] + 0.012], 0.2,
                                    math.degrees(math.atan2(z[1], z[0])) + a.azimuth_offset, -14)
                act = None
                if kind == "pi":
                    act = pi_controller(env, ForcePi())
                elif kind == "armpi":
                    law = ArmAssistPi(push_ramp=0.5, push_speed=0.3, push_ratio=0.8, shift_speed=0.012)
                    law.reset()
                    act = lambda law=law, env=env: law(env)  # noqa: E731
                step, info = 0, {}
                writer.write(overlay.frame(renderer.render(), env, kind=kind, name=name, seed=seed, step=0))
                while True:
                    if act is None:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, te, tr, info = env.step(action)
                    else:
                        obs, _, te, tr, info = env.step_command(act())
                    step += 1
                    rgb = renderer.render()
                    writer.write(overlay.frame(rgb, env, kind=kind, name=name, seed=seed, step=step))
                    if te or tr:
                        break
                m = info.get("episode_metrics") or env.metrics.summary(
                    env.config.contact_threshold, env.penetration_limit)
                still = overlay.frame(rgb, env, kind=kind, name=name, seed=seed, step=step, result=m)
                still.save(a.output.with_name(f"{a.output.stem}_{seed}_{name}.png"))
                for _ in range(a.hold_frames):
                    writer.write(still)
                renderer.close()
                row = {k: m[k] for k in ("success", "crushed", "invalid_contact", "mean_abs_error",
                                         "peak_force_physics", "object_displacement_mm", "goal")}
                row.update(seed=seed, controller=name, object=env.object)
                rows.append(row)
                print(json.dumps(row), flush=True)
        complete = True
    finally:
        writer.close(publish=complete)
        for env in envs.values():
            env.close()
    summary = dict(output=str(a.output), frames=writer.frames, duration_seconds=writer.frames / FPS,
                   seeds=a.seeds, controllers=a.controllers, episodes=rows)
    a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("output", "frames", "duration_seconds")}), file=sys.stderr)


if __name__ == "__main__":
    main()
