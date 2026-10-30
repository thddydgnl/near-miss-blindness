"""Cross-check every number quoted in the JKAIA summary against the frozen result JSON.

Each row: (where it is used, value recomputed from main/results/*.json, string that must appear in the tex).
The script fails loudly when a quoted string is missing from main.tex / main_ko.tex or disagrees with the JSON.
Run from anywhere:  python3 paper/build/verify_numbers.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(B_ROOT, "main", "results")
TEX = {name: open(os.path.join(B_ROOT, "paper", "jkaia2026", name)).read() for name in ("main.tex", "main_ko.tex")}


def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


e1m1 = load("E1_server/m1/e1_m1_analysis.json")
e1m3 = load("E1_server/m3/e1_m3_analysis.json")
m2 = load("E1_m2/analysis/e1_analysis.json")["primary_cell"]["verdict"]
h3m2 = load("E1_m2/analysis/h3_m2.json")["paired"]
e2m1 = load("E2_server/m1/e2_m1_analysis.json")["dp"]
e2m3 = load("E2_server/m3/e2_m3_analysis.json")["verdict_M3"]
e3 = load("E3_server/m1/e3_m1_analysis.json")["verdict_H3"]
e4 = load("E4_server/e4_analysis.json")
e5 = load("E5_server/e5_analysis.json")

h5 = e1m1["cells"]["val/H5/r"]["a"]
v1 = e1m1["verdict_M1"]
m3p = e1m3["pooled"]
a3 = e4["4a_M3"]
b5 = e5["H5b"]["eval"]


def f(x, nd=3):
    """Round half away from zero, as written in the paper."""
    q = 10 ** nd
    return f"{math.copysign(math.floor(abs(x) * q + 0.5) / q, x):.{nd}f}"


def ci(c, nd=3):
    return f"[{f(c[0], nd)}, {f(c[1], nd)}]"


ROWS = [
    # --- abstract and M1 paragraph (val split, H = 5, original r, judge (a) ensemble) ---
    ("M1 AUROC success vs near, real", f(h5["auc_sn_real"]), "0.953"),
    ("M1 AUROC success vs near, imagined", f(h5["auc_sn_imag"]), "0.853"),
    ("M1 AUROC drop near [CI]", f(v1["H1b"]["delta_auc_sn"]) + " " + ci(v1["H1b"]["delta_auc_sn_ci"]), "0.100 [0.075, 0.125]"),
    ("M1 AUROC drop far [CI]", f(v1["H1b"]["delta_auc_sf"]) + " " + ci(v1["H1b"]["delta_auc_sf_ci"]), "0.005 [0.001, 0.010]"),
    ("M1 horizon slope |.| [CI]", f(-v1["H1b"]["horizon_slope"]) + " " + ci([-v1["H1b"]["horizon_slope_ci"][1], -v1["H1b"]["horizon_slope_ci"][0]]), "0.031 [0.025, 0.039]"),
    ("M1 far drop H7, H10", f(e1m1["cells"]["val/H7/r"]["a"]["delta_auc_sf"]) + ", " + f(e1m1["cells"]["val/H10/r"]["a"]["delta_auc_sf"]), "0.021, 0.048"),
    ("M1 far drop range, object-pose r_obj (all H)", f(min(e1m1["cells"][f"val/H{h}/r_obj"]["a"]["delta_auc_sf"] for h in (3, 5, 7, 10))) + "-" + f(max(e1m1["cells"][f"val/H{h}/r_obj"]["a"]["delta_auc_sf"] for h in (3, 5, 7, 10))), "0.058-0.129"),
    ("M1 imagined-feature probe (H4b)", f(e4["4b_M1"]["H4b_M1"]["imag_ceiling"]), "0.889"),
    ("M1 probe drop [CI]", f(e4["4b_M1"]["H4b_M1"]["drop"]) + " " + ci(e4["4b_M1"]["H4b_M1"]["ci_drop"]), "0.064 [0.046, 0.084]"),
    ("M1 success judged success real -> imag", f(h5["p_success_real"]) + " to " + f(h5["p_success_imag"]), "0.862 to 0.710"),
    ("M1 near judged success real -> imag", f(h5["p_near_real"]) + " to " + f(h5["p_near_imag"]), "0.112 to 0.147"),
    ("M1 judge (a) ensemble held-out accuracy", f(load("E1_server/m1/judges_e1.json")["a_full"]["ensemble"]["acc"], 4), "0.9585"),
    ("M2 judge (a) held-out accuracy (E1 mix)", f(load("E1_m2/judge_a/cnn_held_e1.json")["acc_e1_composition"]), "0.974"),
    # --- M3 ---
    ("M3 NM p=0.3/0.1/0.03 (2 dp)", ", ".join(f(a3[k]["NM"], 2) for k in ("p0.3", "p0.1", "p0.03")), "0.10, 0.24, 0.54"),
    ("M3 SM p=0.3/0.1/0.03", ", ".join(f(a3[k]["SM"]) for k in ("p0.3", "p0.1", "p0.03")), "0.021, 0.005, 0.000"),
    ("M3 NM diff 0.03-0.3 [CI]", "+" + f(a3["p0.03_minus_p0.3"]["NM"]["diff"]) + " " + ci(a3["p0.03_minus_p0.3"]["NM"]["ci"]), "+0.431 [0.406, 0.457]"),
    ("M3 SM diff 0.03-0.3 [CI]", f(a3["p0.03_minus_p0.3"]["SM"]["diff"]) + " " + ci(a3["p0.03_minus_p0.3"]["SM"]["ci"]), "-0.021 [-0.030, -0.014]"),
    ("M3 probe ceiling (H4b)", f(e4["4b_M3"]["H4b_M3"]["imag_ceiling"]), "0.986"),
    ("M3 H2a Spearman [CI] (2 dp)", f(e2m3["H2a"]["spearman"], 2) + " " + ci(e2m3["H2a"]["ci"], 2), "0.83 [0.77, 0.94]"),
    ("M3 tied policies real success", f(min(p[2] for p in e2m3["H2b"]["pairs"]), 2) + "-" + f(max(p[3] for p in e2m3["H2b"]["pairs"]), 2), "0.48-0.53"),
    ("M3 tied policies imagined", ", ".join(f(x, 2) for x in sorted(e2m3["H2b"]["tau_densest_window"]["sh"])), "0.58, 0.72, 0.99"),
    ("M3 H2b tau (order kept)", f(e2m3["H2b"]["tau_similar"], 1), "1.0"),
    ("M3 H2a Spearman, design band 1<=r<2 (2 dp)", f(e2m3["design_band"]["spearman"], 2), "0.66"),
    ("M1 fine-tune NM rise only in r_obj cell [CI]", f(e4["4a_M1"]["val_H5_r_obj"]["S97_minus_S50"]["NM"]["diff"]) + " " + ci(e4["4a_M1"]["val_H5_r_obj"]["S97_minus_S50"]["NM"]["ci"]) + " vs r " + ci(e4["4a_M1"]["val_H5_r"]["S97_minus_S50"]["NM"]["ci"]), "0.019 [0.006, 0.031] vs r [-0.006, 0.019]"),
    ("M3 capacity NM small/default/large", ", ".join(f(e4["4b_sweep"][k]["NM"]) for k in ("R3", "base", "R4")), "1.000, 0.535, 0.149"),
    # --- M1 policies (DP 24, H = 5, a_std) ---
    ("M1 DP per-policy o_imag range (2 dp)", f(min(t["o_imag"] for t in e2m1["H5"]["table"]), 2) + " to " + f(max(t["o_imag"] for t in e2m1["H5"]["table"]), 2), "-0.01 to 0.06"),
    ("M1 H2a Spearman (2 dp)", f(e2m1["verdict_M1"]["H2a"]["spearman"], 2), "0.36"),
    # --- table, H3 shares ---
    ("H3 shares M1/M2/M3 (2 dp)", ", ".join([f(e3["M1"]["share"], 2), f(h3m2["share"], 2), f(e3["M3"]["share"], 2)]), "0.24, 0.83, 1.00"),
    ("M2 near judged success real -> imag", f(h3m2["p_real_near"]) + " to " + f(h3m2["p_imag_near"]), "0.051 to 0.298"),
    ("M2 AUROC drop near / far", f(m2["H1b"]["d_auroc_near"]) + " / " + f(m2["H1b"]["d_auroc_far"]), "0.275 / 0.137"),
    ("M3 near judged success imag; AUROC drop near / far", f(m3p["0.03"]["p_near_imag"]) + "; " + f(e1m3["verdict_M3"]["H1b"]["delta_auc_sn"]) + " / " + f(e1m3["verdict_M3"]["H1b"]["delta_auc_sf"]), "0.535; 0.107 / 0.000"),
    # --- remedies ---
    ("H5a M3 (ii)-(i) at p=0.03, 0.1", f(e5["H5a"]["M3_p0.03"]["near_minus_rw"]["NM"]["diff"]) + ", " + f(e5["H5a"]["M3_p0.1"]["near_minus_rw"]["NM"]["diff"]), "0.040, 0.046"),
    ("H5a settings holding", str(e5["H5a"]["verdict"]["settings_holding"]), "0"),
    ("H5b mean |o| det -> MC", f(b5["mean_abs_o"]["deterministic"], 4) + " to " + f(b5["mean_abs_o"]["mc_discount"], 4), "0.0233 to 0.0187"),
    ("H5b diff [CI]", f(b5["diff_mean_abs_o"], 4) + " " + ci(b5["ci_diff_mean_abs_o"], 4), "-0.0046 [-0.0088, -0.0004]"),
    ("H5b FGRE det -> MC", f(b5["fgre"]["deterministic"], 4) + " to " + f(b5["fgre"]["mc_discount"], 4), "0.0225 to 0.0171"),
]

# Strings as they appear in the LaTeX source (minus signs and ranges are typeset differently).
TEX_FORMS = {
    "-0.021 [-0.030, -0.014]": ["$-0.021$ [$-0.030$, $-0.014$]"],
    "-0.0046 [-0.0088, -0.0004]": ["$-0.0046$ [$-0.0088$, $-0.0004$]"],
    "-0.01 to 0.06": ["$-0.01$ to $+0.06$", "$-0.01$~$+0.06$"],
    "+0.431 [0.406, 0.457]": ["$+0.431$ [0.406, 0.457]"],
    "0.48-0.53": ["0.48--0.53", "0.48~0.53"],
    "0.058-0.129": ["0.058--0.129", "0.058~0.129"],
}

def numbers(s):
    for sep in "[],;/":
        s = s.replace(sep, " ")
    return [x.lstrip("+-") for x in s.replace(" to ", " ").split() if any(c.isdigit() for c in x)]


bad = 0
for label, got, want in ROWS:
    ok_value = got == want
    status = "OK " if ok_value else "BAD"
    if not ok_value:
        bad += 1
    print(f"[{status}] {label:48s} json={got:28s} paper={want}")
    if label.startswith("M1 fine-tune NM rise"):
        continue  # backs a sentence without numbers; the value check above is what matters
    for name, text in TEX.items():
        if any(form in text for form in TEX_FORMS.get(want, [want])):
            continue
        # the Korean copy and split sentences quote the same numbers in other words; check them one by one
        missing = [x for x in numbers(want) if x not in text]
        if missing:
            bad += 1
            print(f"        missing in {name}: {missing}")
print("\nall numbers match the JSON" if bad == 0 else f"\n{bad} problem(s)")
sys.exit(1 if bad else 0)
