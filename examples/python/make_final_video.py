"""Assemble the final-assignment video: title card, Pick & Place clip, section
card, grasp-force clip and a closing card (all at 1x speed, 1280x720, 30 fps).

  python examples/python/make_final_video.py --output runs/video/final_video.mp4 \
      --pick-place runs/video/pick_place_demo.mp4 --force-grasp runs/video/force_grasp_demo.mp4

Card texts quote the 200-episode evaluations in docs/PICK_PLACE.md §9 and
docs/FORCE_CONTROL.md §11. Refuses to overwrite the output.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from record_pick_place import FPS, VideoWriter

W, H = 1280, 720
BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
LIGHT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
REPO = "github.com/Natsu-B/Learn-SO-101-basic"

CARDS = {
    "title": [
        ("SO-101による位置可変Pick & Placeと", 46, BOLD, "white"),
        ("柔らかさの分からない物体の把持力制御", 46, BOLD, "white"),
        ("", 20, LIGHT, "white"),
        ("Physical AI 応用編1 最終課題", 28, LIGHT, (190, 200, 214)),
        ("MuJoCoシミュレーション  |  すべて等倍（実時間）で再生", 26, LIGHT, (190, 200, 214)),
    ],
    "part1": [
        ("1. 位置可変Pick & Place", 44, BOLD, (123, 189, 255)),
        ("", 16, LIGHT, "white"),
        ("赤い立方体と緑の目標の位置は、試行ごとに乱数で変わる", 28, LIGHT, "white"),
        ("掴む：IK  →  運ぶ：SAC+HERで学習した方策の推論  →  置く：IK", 28, LIGHT, "white"),
        ("200試行の評価で200試行すべて成功、置いた位置の誤差は平均2.42 mm", 28, LIGHT, (255, 214, 120)),
        ("ここでは6試行を再生する", 24, LIGHT, (190, 200, 214)),
    ],
    "part2": [
        ("2. 柔らかさの分からない物体の把持力制御", 42, BOLD, (72, 224, 170)),
        ("", 16, LIGHT, "white"),
        ("剛性と質量と摩擦は試行ごとに変わり、方策には見せない（接触力を見て握る）", 26, LIGHT, "white"),
        ("SO-101は片側のジョーしか動かず、物体を押し寄せる間に柔らかい物体が潰れる", 26, LIGHT, "white"),
        ("同じ物体で比較：力PI（グリッパのみ）と、腕＋グリッパを学習したPPO", 26, LIGHT, "white"),
        ("200試行の評価で潰さずに目標力で掴めた割合：力PI 16.5 %、PPO 92.0 %", 28, LIGHT, (255, 214, 120)),
    ],
    "end": [
        ("まとめ", 44, BOLD, "white"),
        ("", 16, LIGHT, "white"),
        ("Pick & Place：学習したTransportで200/200成功（位置は毎回ランダム）", 28, LIGHT, "white"),
        ("把持力制御：グリッパ単独の最良19.0 % → 腕＋グリッパのPPOで92.0 %", 28, LIGHT, "white"),
        ("数値はすべて同じseed列の200試行の評価から", 24, LIGHT, (190, 200, 214)),
        ("", 16, LIGHT, "white"),
        (REPO, 28, LIGHT, (123, 189, 255)),
    ],
}


def card(lines):
    image = Image.new("RGB", (W, H), (17, 24, 36))
    d = ImageDraw.Draw(image)
    fonts = [ImageFont.truetype(path, size) for _, size, path, _ in lines]
    heights = [f.size * 1.55 for f in fonts]
    y = (H - sum(heights)) / 2
    for (text, _, _, color), f, h in zip(lines, fonts, heights):
        if text:
            width = d.textlength(text, font=f)
            d.text(((W - width) / 2, y), text, font=f, fill=color)
        y += h
    return image


def write_card(path, lines, seconds):
    writer = VideoWriter(path, W, H)
    image = card(lines)
    image.save(path.with_suffix(".png"))
    try:
        for _ in range(round(seconds * FPS)):
            writer.write(image)
    finally:
        writer.close(publish=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pick-place", type=Path, required=True)
    p.add_argument("--force-grasp", type=Path, required=True)
    p.add_argument("--card-seconds", type=float, default=4.0)
    a = p.parse_args()
    if a.output.exists():
        p.error(f"{a.output} exists; refusing to overwrite")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        p.error("ffmpeg is missing; enter nix develop")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cards = {}
        for name, lines in CARDS.items():
            cards[name] = tmp / f"{name}.mp4"
            write_card(cards[name], lines, a.card_seconds + (1.0 if name == "end" else 0.0))
            shutil.copy(cards[name].with_suffix(".png"), a.output.with_name(f"{a.output.stem}_card_{name}.png"))
        parts = [cards["title"], cards["part1"], a.pick_place, cards["part2"], a.force_grasp, cards["end"]]
        inputs = sum((["-i", str(x)] for x in parts), [])
        streams = "".join(f"[{i}:v]scale={W}:{H},fps={FPS},setsar=1[v{i}];" for i in range(len(parts)))
        concat = "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *inputs,
                        "-filter_complex", streams + concat, "-map", "[out]", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(a.output)], check=True)
    print(a.output)


if __name__ == "__main__":
    main()
