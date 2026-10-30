"""Check the numbers written in content.py against the frozen analysis JSON in main/results.

Each check rounds a JSON value the way the document prints it (negative numbers use an en dash) and confirms that
the printed string appears in the document text. Run after editing content.py:
    python3 paper/explainer/build/verify_numbers.py
"""
import re
import sys

from common import load
from content import DOC

FAILS = []
N_CHECKS = 0


def doc_text():
    parts = []
    for kind, b in DOC:
        if kind in ("p", "h1", "h2"):
            parts.append(b)
        elif kind in ("bullets", "numbers", "formula"):
            parts.extend(b)
        elif kind == "table":
            parts += [b["caption"], b["note"] or ""] + b["header"] + [c for r in b["rows"] for c in r]
        elif kind == "figure":
            parts.append(b["caption"])
        elif kind == "callout":
            parts.append(b["text"])
    text = "\n".join(parts)
    return re.sub(r"\*\*|_\{|\^\{|\}", "", text).replace("\n", " ")


TEXT = doc_text()


def fmt(v, nd, sign=False):
    s = f"{abs(v):.{nd}f}"
    if float(s) == 0:
        return ("+" if sign else "") + s
    return ("–" if v < 0 else ("+" if sign else "")) + s


def chk(label, v, nd, expect=None, sign=False):
    global N_CHECKS
    N_CHECKS += 1
    got = fmt(v, nd, sign)
    exp = expect if expect is not None else got
    if got != exp:
        FAILS.append(f"{label}: JSON gives {got}, document expects {exp}")
    elif exp not in TEXT:
        FAILS.append(f"{label}: '{exp}' not found in the document")


def ci(label, lo, hi, nd, expect=None):
    global N_CHECKS
    N_CHECKS += 1
    got = f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"
    exp = expect or got
    if got != exp:
        FAILS.append(f"{label}: JSON gives {got}, document expects {exp}")
    elif exp not in TEXT:
        FAILS.append(f"{label}: '{exp}' not found in the document")


def main():
    # ---- M1 curves (E1, val split, judge (a))
    e1 = load("E1_server/m1/e1_m1_analysis.json")["cells"]
    a5 = e1["val/H5/r"]["a"]
    chk("M1 AUROC near real H5", a5["auc_sn_real"], 3, "0.953")
    chk("M1 AUROC near imag H5", a5["auc_sn_imag"], 3, "0.853")
    chk("M1 dAUROC near", a5["delta_auc_sn"], 3, "0.100")
    ci("M1 dAUROC near CI", *a5["ci"]["delta_auc_sn"], 3, "[0.075, 0.125]")
    chk("M1 dAUROC far", a5["delta_auc_sf"], 3, "0.005")
    ci("M1 dAUROC far CI", *a5["ci"]["delta_auc_sf"], 3, "[0.001, 0.010]")
    chk("M1 AUROC far imag", a5["auc_sf_imag"], 3, "0.995")
    chk("M1 floor", a5["p_near_real"], 3, "0.112")
    chk("M1 near imag", a5["p_near_imag"], 3, "0.147")
    chk("M1 succ real", a5["p_success_real"], 3, "0.862")
    chk("M1 succ imag", a5["p_success_imag"], 3, "0.710")
    chk("M1 r50 real", a5["r50_real"], 2, "1.03")
    chk("M1 r50 imag", a5["r50_imag"], 2, "0.90")
    chk("M1 dr10", a5["delta_r10"], 2, "0.21")
    chk("M1 ceiling", a5["auc_sn_ceiling"], 3, "0.889")
    hs = e1["val/r/horizon_slope_a"]
    chk("M1 horizon slope", -hs["slope"], 3, "0.031")
    ci("M1 horizon slope CI", -hs["ci"][1], -hs["ci"][0], 3, "[0.025, 0.039]")
    rows = {3: ("0.110", "0.798", "0.145", "0.953", "0.900", "0.906", "0.053", "0.001"),
            7: ("0.100", "0.582", "0.172", "0.950", "0.767", "0.828", "0.183", "0.021"),
            10: ("0.107", "0.455", "0.204", "0.939", "0.673", "0.725", "0.266", "0.048")}
    for h, exp in rows.items():
        c = e1[f"val/H{h}/r"]["a"]
        keys = ("p_near_real", "p_success_imag", "p_near_imag", "auc_sn_real", "auc_sn_imag", "auc_sn_ceiling",
                "delta_auc_sn", "delta_auc_sf")
        for k, e in zip(keys, exp):
            chk(f"M1 H{h} {k}", c[k], 3, e)
    floors = [e1[f"val/H{h}/r_obj"]["a"]["p_near_real"] for h in (3, 5, 7, 10)]
    chk("M1 r_obj floor min", min(floors), 3, "0.044")
    chk("M1 r_obj floor max", max(floors), 3, "0.050")
    chk("M1 r_obj far H3", e1["val/H3/r_obj"]["a"]["p_far_imag"], 3, "0.113")
    dfar = [e1[f"val/H{h}/r_obj"]["a"]["delta_auc_sf"] for h in (3, 5, 7, 10)]
    chk("M1 r_obj dAUROC far min", min(dfar), 3, "0.058")
    chk("M1 r_obj dAUROC far max", max(dfar), 3, "0.129")
    comp = a5["near_composition"]
    if f"{comp['agent']['n']:,}" not in TEXT or f"{a5['n']['near']:,}" not in TEXT:
        FAILS.append("M1 near composition counts not found")

    # ---- H4b (E4b) and M1 fine-tuning (E4a)
    e4 = load("E4_server/e4_analysis.json")
    b = e4["4b_M1"]["val_H5_r"]
    chk("H4b M1 drop", b["drop"], 3, "0.064")
    ci("H4b M1 drop CI", *b["ci_drop"], 3, "[0.046, 0.084]")
    d = e4["4a_M1"]["val_H5_r"]["S97_minus_S50"]
    chk("E4a M1 NM", d["NM"]["diff"], 3, "+0.007", sign=True)
    ci("E4a M1 NM CI", *d["NM"]["ci"], 3, "[–0.006, 0.019]")
    chk("E4a M1 SM", d["SM"]["diff"], 3, "–0.029")
    ci("E4a M1 SM CI", *d["SM"]["ci"], 3, "[–0.053, –0.006]")
    d = e4["4a_M1"]["val_H5_r_obj"]["S97_minus_S50"]
    chk("E4a M1 r_obj NM", d["NM"]["diff"], 3, "+0.019", sign=True)
    chk("E4a M1 r_obj SM", d["SM"]["diff"], 3, "–0.037")

    # ---- M3 (E1 pooled, E4a)
    m3 = load("E1_server/m3/e1_m3_analysis.json")["pooled"]
    for p, nm, r29, auc, ceil, r50, dr10 in (("0.3", "0.104", "0.452", "0.966", "0.990", "1.06", "0.23"),
                                             ("0.1", "0.238", "0.807", "0.930", "0.988", "1.20", "0.42"),
                                             ("0.03", "0.535", "0.979", "0.893", "0.986", "1.51", "0.95")):
        c = m3[p]
        chk(f"M3 p{p} near", c["p_near_imag"], 3, nm)
        chk(f"M3 p{p} r29", c["p_near_r29_imag"], 3, r29)
        chk(f"M3 p{p} auc imag", c["auc_sn_imag"], 3, auc)
        chk(f"M3 p{p} ceiling", c["auc_sn_ceil"], 3, ceil)
        chk(f"M3 p{p} r50", c["r50_imag"], 2, r50)
        chk(f"M3 p{p} dr10", c["delta_r10"], 2, dr10)
    chk("M3 far (as percent)", 100 * m3["0.03"]["p_far_imag"], 1, "1.0")
    chk("M3 near/far", m3["0.03"]["ratio_near_far_imag"], 1, "53.5")
    chk("M3 dAUROC near", m3["0.03"]["delta_auc_sn"], 3, "0.107")
    a = e4["4a_M3"]
    for p, sm in (("0.3", "0.0215"), ("0.1", "0.0046"), ("0.03", "0.000")):
        chk(f"M3 SM p{p}", a[f"p{p}"]["SM"], len(sm) - 2, sm)
    dd = a["p0.03_minus_p0.3"]
    chk("H4a M3 NM", dd["NM"]["diff"], 3, "+0.431", sign=True)
    ci("H4a M3 NM CI", *dd["NM"]["ci"], 3, "[0.406, 0.457]")
    chk("H4a M3 SM", dd["SM"]["diff"], 4, "–0.0215")
    ci("H4a M3 SM CI", *dd["SM"]["ci"], 3, "[–0.030, –0.014]")
    chk("H4b M3 gap", e4["4b_M3"]["p0.03"]["drop"], 3, "0.014")

    # ---- M2 (E1 RunPod)
    m2 = load("E1_m2/analysis/e1_analysis.json")["primary_cell"]["verdict"]
    h1a, h1b = m2["H1a"], m2["H1b"]
    chk("M2 slope", h1a["slope"], 2, "–1.44")
    ci("M2 slope CI", *h1a["slope_ci95"], 2, "[–1.82, –1.12]")
    chk("M2 near", h1a["near"], 3, "0.298")
    ci("M2 near CI", *h1a["near_ci95"], 3, "[0.241, 0.355]")
    chk("M2 far", h1a["far"], 3, "0.118")
    chk("M2 near/far", h1a["near_over_far"], 2, "2.52")
    ci("M2 near/far CI", *h1a["near_over_far_ci95"], 2, "[1.79, 3.95]")
    chk("M2 dAUROC near", h1b["d_auroc_near"], 3, "0.275")
    ci("M2 dAUROC near CI", *h1b["d_auroc_near_ci95"], 3, "[0.240, 0.311]")
    chk("M2 dAUROC far", h1b["d_auroc_far"], 3, "0.137")
    ci("M2 dAUROC far CI", *h1b["d_auroc_far_ci95"], 3, "[0.104, 0.173]")
    chk("M2 k slope", h1b["horizon_slope"], 3, "–0.065")
    ci("M2 k slope CI", *h1b["horizon_slope_ci95"], 3, "[–0.077, –0.052]")
    pt = load("E1_m2/analysis/e1_analysis.json")["cells"]["a|pooled6"]["point"]
    chk("M2 success judged imag", pt["k1"]["success"], 3, "0.588")
    chk("M2 floor", pt["real"]["near"], 3, "0.051")

    # ---- H3 decomposition (E3, h3_m2)
    e3 = load("E3_server/m1/e3_m1_analysis.json")
    for label, c, share, cis in (("H3 M1 r", e3["e1_cells"]["val_H5_r"]["a"], "0.239", "[0.067, 0.389]"),
                                 ("H3 M1 r_obj", e3["e1_cells"]["val_H5_r_obj"]["a"], "0.765", "[0.697, 0.822]"),
                                 ("H3 M1 DP", e3["dp"]["H5"]["a_std"], "0.698", "[0.554, 0.826]"),
                                 ("H3 M1 VLM", e3["e1_cells"]["val_H5_r"]["d"], "–0.020", "[–0.043, 0.000]")):
        chk(label + " share", c["share"], 3, share)
        ci(label + " CI", *c["ci_share"], 3, cis)
    chk("VLM floor r", e3["e1_cells"]["val_H5_r"]["d"]["p_near_real"], 3, "0.893")
    chk("VLM imag r", e3["e1_cells"]["val_H5_r"]["d"]["p_near_imag"], 3, "0.876")
    h3 = load("E1_m2/analysis/h3_m2.json")["paired"]
    chk("H3 M2 share", h3["share"], 3, "0.829")
    ci("H3 M2 CI", *h3["share_ci95"], 3, "[0.754, 0.886]")

    # ---- policy level (E2)
    e2 = load("E2_server/m1/e2_m1_analysis.json")
    dp = e2["dp"]["H5"]
    chk("H2a M1", dp["point"]["spearman_n_oimag"], 3, "0.357")
    ci("H2a M1 CI", *dp["ci"]["spearman_n_oimag"], 3, "[0.023, 0.598]")
    chk("H2a M1 perm p", dp["perm_spearman_n_oimag"]["p_one_sided"], 3, "0.047")
    chk("H2b M1 tau", dp["point"]["tau_similar"], 3, "0.709")
    chk("H2c M1 pearson", dp["point"]["pearson_sh_s"], 3, "0.998")
    chk("H2c M1 mmrv", dp["point"]["mmrv"], 3, "0.010")
    chk("M1 FGRE", dp["point"]["fgre"], 3, "0.023")
    chk("M1 FGRE real judge", dp["point"]["fgre_realjudge"], 3, "0.005")
    chk("M1 MMRV real judge", dp["point"]["mmrv_realjudge"], 3, "0.001")
    chk("M1 tau real judge", dp["point"]["tau_similar_realjudge"], 3, "0.911")
    chk("M1 max gap similar", dp["point"]["h2b_max_gap_similar"], 3, "0.060")
    if f"{int(dp['point']['n_similar_pairs'])}쌍" not in TEXT:
        FAILS.append("M1 similar pair count not found")
    pooled = dp["pooled"]
    chk("DP near real", pooled["p_judged_success_given_near_real"], 3, "0.058")
    chk("DP near imag", pooled["p_judged_success_given_near_imag"], 3, "0.191")
    chk("DP succ real", pooled["p_judged_success_given_success_real"], 3, "0.993")
    chk("DP succ imag", pooled["p_judged_success_given_success_imag"], 3, "0.990")
    ctrl = e2["ctrl"]["a"]["point"]
    chk("ctrl FGRE", ctrl["fgre"], 3, "0.117")
    chk("ctrl MMRV", ctrl["mmrv"], 3, "0.023")
    m3p = load("E2_server/m3/e2_m3_analysis.json")
    r29 = m3p["0.03"]["r29"]
    chk("H2a M3", r29["point"]["spearman_n_oimag"], 3, "0.829")
    ci("H2a M3 CI", *r29["ci"]["spearman_n_oimag"], 3, "[0.771, 0.943]")
    chk("H2a M3 perm p", r29["perm_spearman_n_oimag"]["p_one_sided"], 3, "0.029")
    chk("H2c M3 pearson", r29["point"]["pearson_sh_s"], 3, "0.694")
    chk("H2c M3 mmrv", r29["point"]["mmrv"], 3, "0.157")
    chk("M3 FGRE", r29["point"]["fgre"], 3, "0.000")
    chk("H2a M3 design band", m3p["verdict_M3"]["design_band"]["spearman"], 2, "0.66")
    expect = {"pi1_precise": ("0.997", "0.002", "1.000", "0.003"), "pi2_good": ("0.901", "0.054", "0.982", "0.081"),
              "pi3_medium": ("0.716", "0.081", "0.886", "0.171"), "pi4_sloppy": ("0.521", "0.088", "0.722", "0.202"),
              "pi5_far_failer": ("0.482", "0.000", "0.580", "0.098"),
              "pi6_near_failer": ("0.526", "0.474", "0.985", "0.459")}
    for row in r29["table"]:
        for k, e in zip(("s", "n", "sh", "o_imag"), expect[row["policy"]]):
            chk(f"M3 {row['policy']} {k}", row[k], 3, e)

    # ---- prescriptions (E5)
    e5 = load("E5_server/e5_analysis.json")
    h5a = e5["H5a"]
    nm = {"M1": ("0.151", "0.143", "0.143", "0.149", "0.155", "0.144"),
          "M3_p0.03": ("0.535", "0.187", "0.227", "0.197", "0.210", "0.201"),
          "M3_p0.1": ("0.238", "0.142", "0.188", "0.156", "0.159", "0.164")}
    sm = {"M1": ("0.230", "0.236", "0.242", "0.246", "0.239", "0.243"),
          "M3_p0.03": ("0.000", "0.007", "0.010", "0.016", "0.006", "0.007"),
          "M3_p0.1": ("0.005", "0.014", "0.014", "0.024", "0.014", "0.016")}
    for s in nm:
        for k, n_e, s_e in zip(("ref", "rw", "near", "cp", "full", "bw"), nm[s], sm[s]):
            chk(f"H5a {s} {k} NM", h5a[s][k]["NM"], 3, n_e)
            chk(f"H5a {s} {k} SM", h5a[s][k]["SM"], 3, s_e)
    for s, dv, dci in (("M1", "+0.001", "[–0.011, 0.014]"), ("M3_p0.03", "+0.040", "[0.017, 0.064]"),
                       ("M3_p0.1", "+0.046", "[0.028, 0.065]")):
        chk(f"H5a {s} near-rw", h5a[s]["near_minus_rw"]["NM"]["diff"], 3, dv, sign=True)
        ci(f"H5a {s} near-rw CI", *h5a[s]["near_minus_rw"]["NM"]["ci"], 3, dci)
    ev = e5["H5b"]["eval"]
    res = ev["results"]
    chk("H5b det |o|", res["deterministic"]["point"]["mean_abs_o"], 4, "0.0233")
    chk("H5b mc |o|", res["mc_discount"]["point"]["mean_abs_o"], 4, "0.0187")
    chk("H5b maj |o|", res["mc_majority"]["point"]["mean_abs_o"], 4, "0.0283")
    chk("H5b diff", ev["diff_mean_abs_o"], 4, "–0.0046")
    ci("H5b diff CI", *ev["ci_diff_mean_abs_o"], 4, "[–0.0088, –0.0004]")
    chk("H5b det FGRE", res["deterministic"]["point"]["fgre"], 4, "0.0225")
    chk("H5b mc FGRE", res["mc_discount"]["point"]["fgre"], 4, "0.0171")
    chk("H5b maj FGRE", res["mc_majority"]["point"]["fgre"], 4, "0.0254")
    chk("H5b mc spearman", res["mc_discount"]["point"]["spearman_n_oimag"], 3, "0.020")
    chk("H5b maj spearman", res["mc_majority"]["point"]["spearman_n_oimag"], 3, "0.449")
    chk("H5b mean o det", res["deterministic"]["point"]["mean_o"], 3, "+0.023", sign=True)
    chk("H5b mean o mc", res["mc_discount"]["point"]["mean_o"], 3, "+0.014", sign=True)
    chk("H5b mean o maj", res["mc_majority"]["point"]["mean_o"], 3, "+0.028", sign=True)

    # ---- capacity sweep and loss shares (E4b sweep, E4c)
    sw = e4["4b_sweep"]
    for k, e in (("R3", "1.000"), ("base", "0.535"), ("R4", "0.149"), ("R1", "0.614"), ("R2", "0.653")):
        chk(f"sweep {k} NM", sw[k]["NM"], 3, e)
    chk("sweep R4 auc", sw["R4"]["auc_sn_imag"], 3, "0.956")
    chk("sweep R2 SM", sw["R2"]["SM"], 3, "0.264")
    ceil = [sw[k]["auc_sn_ceiling"] for k in ("base", "R1", "R3", "R4")]
    chk("sweep ceiling min", min(ceil), 3, "0.967")
    chk("sweep ceiling max", max(ceil), 3, "0.993")
    chk("E4c M1 block", e4["4c_M1"]["innovation_energy_share"]["block"], 3, "0.189")
    chk("E4c M1 background", e4["4c_M1"]["innovation_energy_share"]["background"], 3, "0.727")
    chk("E4c M3 object", e4["4c_M3"]["innovation_energy_share"]["obj"], 3, "0.166")

    print(f"{N_CHECKS} checks, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAIL", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
