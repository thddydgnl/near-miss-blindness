"""Freeze E1 severity settings from the E0b calibration (standard library only).
For each task's boundary-targeted family: isotonic (non-decreasing) fit of median settled r vs severity on the
calibration demos d00-d09, then inverse interpolation at 15 target r values evenly spaced in [0.85, 2.95]
(targets above the curve's maximum map to the largest calibrated severity). CureWM's six families keep their own
grid (0.2, 0.4, 0.6, 0.8, 1.0). Far-miss candidates (r >= 3, needed for the near/far contrast of H1a): the largest
calibrated severity and 1.25x it (capped at 1.0 for under_actuation / pull_truncate).
Output: {task: {family, severities, far_severities, curewm_grid}} for make_manifest.py e1-candidates.
Usage: python3 freeze_severity.py <calib.json> <out.json>
"""
import json
import sys

calib = json.load(open(sys.argv[1]))
TARGETS = [0.85 + i * (2.95 - 0.85) / 14 for i in range(15)]
B_FAM = {"place_offset", "under_actuation", "pull_truncate"}
out = {}
for task, ti in calib["tasks"].items():
    fam = [f for f in ti["families"] if f in B_FAM][0]
    pts = sorted((float(s), m) for s, m in ti["families"][fam]["median_r_by_severity"].items())
    # pool-adjacent-violators for a non-decreasing fit
    blocks = [[m, 1, [s]] for s, m in pts]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = [(s, b[0]) for b in blocks for s in b[2]]
    sev = []
    for r in TARGETS:
        if r >= fit[-1][1]:
            sev.append(fit[-1][0])
            continue
        if r <= fit[0][1]:
            sev.append(fit[0][0])
            continue
        for (s0, r0), (s1, r1) in zip(fit[:-1], fit[1:]):
            if r0 <= r <= r1:
                sev.append(s0 + (s1 - s0) * ((r - r0) / (r1 - r0) if r1 > r0 else 0.0))
                break
    top = pts[-1][0]
    far = sorted({round(top, 4), round(min(1.25 * top, 1.0) if fam != "place_offset" else 1.25 * top, 4)})
    out[task] = dict(family=fam, severities=sorted({round(s, 4) for s in sev}), far_severities=far,
                     targets=[round(t, 3) for t in TARGETS],
                     isotonic_fit=[(round(s, 4), round(r, 3)) for s, r in fit], curewm_grid=[0.2, 0.4, 0.6, 0.8, 1.0] if task != "0" else [])
json.dump(out, open(sys.argv[2], "w"), indent=1)
for t, v in out.items():
    print(t, v["family"], v["severities"])
