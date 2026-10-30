"""Offscreen rendering is read-only; encoded frames match actual control steps."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from robot_sim import PickPlaceRenderer, PickPlaceSimulator
from robot_sim.pick_place_pipeline import run_pick_place_episode


has_display = sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))


@pytest.mark.skipif(not has_display, reason="GLFW requires a display server")
def test_render_rgb_and_pipeline_physics_are_unchanged():
    sim = PickPlaceSimulator()
    renderer = None
    try:
        reference = sim.run_episode(25)
        sim.reset(25)
        renderer = PickPlaceRenderer(sim, 640, 360)
        initial = sim.state()
        first = renderer.render()
        assert len(first) == 640 * 360 * 3
        pixels = np.frombuffer(first, dtype=np.uint8).reshape(360, 640, 3)
        assert pixels.std() > 10
        assert renderer.render() == first
        assert sim.state() == initial
        sim.reset(99)
        assert renderer.render() != first  # cube and mocap goal move with the seed
        frames = {}

        def on_step(state, name, learned):
            if name not in frames:
                before = sim.state()
                predicates = sim.predicates()
                report = sim.skill_report()
                frames[name] = renderer.render()
                assert sim.state() == before == state
                assert sim.predicates() == predicates
                assert sim.skill_report() == report
                assert not learned

        actual = run_pick_place_episode(25, sim=sim, transport="scripted", on_step=on_step)
        assert {k: actual[k] for k in reference} == reference
        assert actual["success"]
        assert set(frames) == {"grasp", "transport", "place"}
        assert len(set(frames.values())) == 3
        renderer.close()
        renderer.close()
        with pytest.raises(ValueError, match="closed"):
            renderer.render()
        with pytest.raises(ValueError, match="render size"):
            PickPlaceRenderer(sim, 0, 360)
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()


@pytest.mark.skipif(not has_display or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="requires GLFW display, ffmpeg and ffprobe")
def test_record_cli_encodes_control_steps_and_holds(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples/python/record_pick_place.py"
    output = tmp_path / "demo.mp4"
    result = subprocess.run([
        sys.executable, str(script), "--transport", "scripted", "--seeds", "0",
        "--width", "640", "--height", "360", "--hold-frames", "2", "--output", str(output),
    ], check=True, capture_output=True, text=True, timeout=90)
    row = json.loads(result.stdout)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert row["success"] and summary["success_rate"] == 1
    assert summary["frames"] == row["control_steps"] + 2
    assert summary["learned_skills"] == []
    assert summary["file_size_bytes"] == output.stat().st_size
    assert json.loads(output.with_suffix(".jsonl").read_text()) == row
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(output),
    ], text=True))["streams"][0]
    assert (probe["width"], probe["height"]) == (640, 360)
    assert probe["r_frame_rate"] == "30/1"
    assert int(probe["nb_frames"]) == summary["frames"]
    for skill in ("grasp", "transport", "place"):
        assert output.with_name(f"demo_{skill}.png").is_file()
