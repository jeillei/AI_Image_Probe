"""Feature-family taxonomy for SynthImage Representation v1 (SD1.5 rich signature).

Families are assigned by column-name pattern only; no data are consulted, so the
taxonomy cannot be tuned to any result.  Every v1 column maps to exactly one family.
"""
from __future__ import annotations
import re
# ordered: first match wins
_RULES = [
 ("roundtrip",  r"^(pixel_roundtrip_mse|latent_roundtrip_mse)$"),
 ("endpoint",   r"^(endpoint_)"),
 ("legacy_geometry", r"^(path_length|speed_mean|speed_max|acceleration_abs_mean)$"),
 ("legacy_noise", r"^eps_"),
 ("legacy_guidance", r"^guidance_"),
 ("rich_score_spatial", r"^rich_score_t\d+_(channel_|h_autocorr|v_autocorr)"),
 ("rich_score_fft", r"^rich_score_t\d+_fft_"),
 ("rich_score_moments", r"^rich_score_t\d+_(mean|var|skew|kurtosis|q\d+|median|absmean|rms|maxabs|nearzero)$"),
 ("rich_guidance_moments", r"^rich_guidance_t\d+_"),
 ("rich_latent_typicality", r"^rich_latent_t\d+_"),
 ("rich_crosstime", r"^rich_(score_crosscos|dz_crosscos)_"),
 ("rich_score_dynamics", r"^rich_score_(norm|change|consecutive_cos|angle|relchange|secondchange|first_last_cos|cumulative_angle)"),
 ("rich_latent_geometry", r"^rich_latent_(step|turn|accel|path|endpoint|cumulative)"),
 ("rich_alignment", r"^rich_(cond_dz|uncond_dz|guidance_dz|cond_uncond|guidance_cond)_"),
]
def family(col: str) -> str:
    for name, pat in _RULES:
        if re.search(pat, col): return name
    return "unassigned"
# coarse groups requested in the task spec
COARSE = {
 "legacy51": ["roundtrip","endpoint","legacy_geometry","legacy_noise","legacy_guidance"],
 "reconstruction_roundtrip": ["roundtrip"],
 "trajectory_noise": ["legacy_noise","rich_score_dynamics"],
 "guidance": ["legacy_guidance","rich_guidance_moments"],
 "latent_endpoint_stats": ["endpoint","rich_latent_typicality"],
 "spatial": ["rich_score_spatial"],
 "fft": ["rich_score_fft"],
 "crosstime": ["rich_crosstime"],
 "alignment": ["rich_alignment"],
 "latent_geometry": ["legacy_geometry","rich_latent_geometry"],
 "score_moments": ["rich_score_moments"],
}
