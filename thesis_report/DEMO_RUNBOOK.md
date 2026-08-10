# Demo runbook

Every command below was run and timed on this machine at 00:20 on 10 August.
Timings are cold-start on CPU.

**Always from the repository root:** `E:\thesis\MotionAGFormer`

```
cd E:\thesis\MotionAGFormer
```

---

## 1. The Gradio app (the interactive demo)

```
python app.py
```

It prints a local URL — open it in a browser. This is the interface on slide 5
of the report (Figure 5.2), with the **Representation** control that switches
between the camera frame and the body frame.

> **Start it before the viva begins**, not during. The first load pulls the
> detector and the lifter into memory and you do not want to spend that silence
> in front of the panel. Leave the tab open.

---

## 2. A single image — 16 seconds

```
python demo_live/infer_image.py --input "examples/Screenshot 2026-07-20 223512.png" --output demo_out.png
```

Prints four steps and writes `demo_out.png`. Any `.jpg` or `.png` with one clearly
visible person works. Useful flags:

| Flag | Default | Use |
|---|---|---|
| `--conf` | 0.4 | lower it to `0.25` if the detector misses the person |
| `--det-weights` | `yolov8n-pose.pt` | `yolov8m-pose.pt` is slower and more accurate |
| `--device` | `cpu` | leave it |

## 3. A video

```
python demo_live/infer_video.py --input demo/video/person.mp4 --output demo_out.mp4
```

Roughly **8 minutes for a 30-second clip on CPU.** Do not run this live. Render
anything you want to show in advance.

Available clips: `demo/video/person.mp4`, `demo/video/sample_video.mp4`, and
several in `examples/`.

---

## 4. The eight-camera cross-view figure — 29 seconds

```
python -m evaluation.make_realview_figure --cams 8
```

Prints:

```
raw 0.1491 -> canonical 0.0946 (normalised units, -36.6%)
```

and writes `thesis_artifacts/figures/fig_realview.png`, copying it into
`thesis_report/images/`. **This is the strongest thing you can run live** — it
regenerates the figure on slide 4 from the cached predictions in about half a
minute, which demonstrates that the figure is not hand-made.

Options: `--cams 2` … `--cams 8`, `--subject S1|S2`, `--dynamic` for the walking
window instead of the static one, `--out somename.png`.

## 5. The animation on slide 5

```
python -m evaluation.make_realview_animation
```

About two minutes; writes `anim_realview.mp4` and `.gif` into
`thesis_report/images/`. Already rendered — only re-run if asked to prove it.

---

## 6. If someone asks you to prove the numbers

```
python -m evaluation.audit_numbers          # 304/304 claims re-derived
python -m evaluation.verify_prereg_order    # 16/16, each criterion precedes its result
python -m unittest discover -s tests -q     # 76 tests
```

`audit_numbers` takes a few seconds and prints every claim with its source file.
`verify_prereg_order` reads the git history and prints the gap in minutes between
each pre-registration commit and its result.

---

## Before you leave the house

- [ ] `python app.py` starts and the browser tab opens
- [ ] `demo_out.png` still on disk, in case the live run fails
- [ ] `fig_realview.png` and `anim_realview.mp4` present in `thesis_report/images/`
- [ ] `Thesis_12108004.pdf` on a USB stick **and** emailed to yourself
- [ ] `DEFENSE_SLIDES.html` exported to PDF as a fallback — if the projector will
      not play video, the poster frame still shows

**If a live command fails, do not debug it in the room.** Say *"I have the output
here"*, show the saved file, and offer to run it afterwards. A failed live demo
costs nothing; five minutes of silent debugging costs a lot.
