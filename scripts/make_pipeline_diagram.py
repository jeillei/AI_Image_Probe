"""Generate the project's compact pipeline diagram for the README / docs. Publication-quality, code-generated
(no external design tool), matches the frozen probe structure in src/probes/sd15.py and the v2 panel in
src/features/panel_v2.py exactly -- this is a documentation aid, not a new artifact with its own claims."""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("results/summary"); OUT.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("image", "real photograph\nor generated image", "#888888"),
    ("VAE", "lpips_ae\npixel_mse_ae\nlatent_mse_ae", "#4477aa"),
    ("score\nresponse", "lare_t200\nscore_norm_step0", "#66aa44"),
    ("inverse\ntrajectory", "diffpath_curvature\npath_length", "#aa3377"),
    ("round\ntrip", "pixel_l1_roundtrip\nlpips_roundtrip\nlatent_mse_roundtrip", "#cc6633"),
]

fig, ax = plt.subplots(figsize=(13, 3.6))
ax.set_xlim(0, len(STAGES)); ax.set_ylim(0, 1); ax.axis("off")

box_w, box_h = 0.82, 0.5
for i, (title, feats, color) in enumerate(STAGES):
    x = i + (1 - box_w) / 2
    y = 0.28
    box = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.6, edgecolor=color, facecolor=color + "22")
    ax.add_patch(box)
    ax.text(i + 0.5, y + box_h - 0.1, title, ha="center", va="top", fontsize=12, fontweight="bold", color=color)
    ax.text(i + 0.5, y + box_h / 2 - 0.08, feats, ha="center", va="center", fontsize=8.3, color="#222222")
    if i > 0:
        arrow = FancyArrowPatch((i - (1 - box_w) / 2 - 0.02, y + box_h / 2), (x + 0.02, y + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#444444")
        ax.add_patch(arrow)

ax.text(len(STAGES) / 2, 0.06, "same frozen SD1.5 probe, unmodified, for every generator tested "
        "(SD1.5 / SDXL / PixArt-Sigma DiT / aMUSEd)", ha="center", va="center", fontsize=9.5, style="italic", color="#555555")
ax.text(len(STAGES) / 2, 0.97, "SynthImage frozen v2 stage panel  —  where does forensic information enter the pipeline?",
        ha="center", va="top", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "pipeline_diagram.png", dpi=180, bbox_inches="tight")
print("saved", OUT / "pipeline_diagram.png")
