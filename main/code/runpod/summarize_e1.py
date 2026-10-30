"""Markdown tables from analyze_e1.py output (standard library only; runs on the Mac).
Usage: python3 summarize_e1.py <e1_analysis.json> > tables.md"""
import json
import sys

d = json.load(open(sys.argv[1]))
cells = d["cells"]


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, str):
        return "∞" if x == "inf" else x
    return f"{x:.{nd}f}"


def ci(c, key, nd=3, holm=False):
    v = c["ci"].get(key)
    if not v:
        return ""
    lo, hi = (v[0], v[3]) if holm else (v[1], v[2])
    return f" [{f(lo, nd)}, {f(hi, nd)}]"


def get(p, path):
    for k in path.split("."):
        if p is None:
            return None
        p = p.get(k)
    return p


prim = d["primary_judge"]
print(f"## primary cell {prim}|pooled6, k = 1\n")
v = d["primary_cell"]["verdict"]
c = cells[f"{prim}|pooled6"]
print("| item | value [95% CI] | criterion | met |\n|---|---|---|---|")
print(f"| H1a(i) slope on log r, failures r>=1 | {f(get(c['point'], 'k1.slope_fail'))}{ci(c, 'k1.slope_fail')} (Holm{ci(c, 'k1.slope_fail', holm=True)}) | CI < 0 | {v['H1a']['i_monotone']} |")
print(f"| H1a(ii) P(S=1 given near) | {f(get(c['point'], 'k1.near'))}{ci(c, 'k1.near')} | >= 0.3 | {v['H1a']['ii_near>=0.3']} |")
print(f"| P(S=1 given far) | {f(get(c['point'], 'k1.far'))}{ci(c, 'k1.far')} | | |")
print(f"| H1a(iii) near/far | {f(get(c['point'], 'k1.near_over_far'), 2)}{ci(c, 'k1.near_over_far', 2)} | >= 3 | {v['H1a']['iii_near/far>=3']} |")
print(f"| **H1a** | | | **{v['H1a']['holds']}** |")
print(f"| H1b(i) dAUROC success vs near (real - imag) | {f(get(c['point'], 'k1.d_auroc_near'))}{ci(c, 'k1.d_auroc_near')} | >= 0.05 | {v['H1b']['i_d_near>=0.05']} |")
print(f"| H1b(ii) dAUROC success vs far | {f(get(c['point'], 'k1.d_auroc_far'))}{ci(c, 'k1.d_auroc_far')} | <= 0.02 | {v['H1b']['ii_d_far<=0.02']} |")
print(f"| H1b(iii) slope over k of [imag - real] (near) | {f(c['point'].get('horizon_slope_near'), 4)}{ci(c, 'horizon_slope_near', 4)} (Holm{ci(c, 'horizon_slope_near', 4, holm=True)}) | CI < 0 | {v['H1b']['iii_slope_ci<0']} |")
print(f"| **H1b** | | | **{v['H1b']['holds']}** |")
print(f"| **go** (H1a or H1b) | | | **{v['go']}** |\n")

print("## composition\n")
print("| group | n | start states | success | near | mid | far | fail r<1 |\n|---|---|---|---|---|---|---|---|")
for g, x in d["composition"].items():
    print(f"| {g} | {x['n']} | {x['start_states']} | {x['success']} | {x['near']} | {x['mid']} | {x['far']} | {x['fail_r<1']} |")
print()

for g in ["pooled6", "t0"]:
    print(f"## {g}: band probabilities P(S_hat = 1), AUROC, curve summaries\n")
    print("| judge | frames | success | near | mid | far | near/far | slope (fail) | AUROC s-vs-near | AUROC s-vs-far | dAUROC near | dAUROC far | r50 | r10 | dr10 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in ("a", "c", "c_lift", "d"):
        c = cells.get(f"{j}|{g}")
        if c is None:
            continue
        p = c["point"]
        if j != "c_lift":
            r = p["real"]
            print(f"| {j} | real | {f(r['success'])} | {f(r['near'])}{ci(c, 'real.near')} | {f(r['mid'])} | {f(r['far'])} | {f(r['near_over_far'], 2)} | {f(r['slope_fail'], 2)} | {f(r['auroc_near'])}{ci(c, 'real.auroc_near')} | {f(r['auroc_far'])} | | | {f(r['curve']['r50'], 2)} | {f(r['curve']['r10'], 2)} | |")
        for k in (1, 2, 3, 4):
            q = p.get(f"k{k}")
            if q is None:
                continue
            print(f"| {j} | imag k={k} | {f(q['success'])} | {f(q['near'])}{ci(c, f'k{k}.near')} | {f(q['mid'])} | {f(q['far'])} | {f(q['near_over_far'], 2)} | {f(q['slope_fail'], 2)}{ci(c, f'k{k}.slope_fail', 2)} | {f(q['auroc_near'])}{ci(c, f'k{k}.auroc_near')} | {f(q['auroc_far'])}{ci(c, f'k{k}.auroc_far')} | {f(q['d_auroc_near'])}{ci(c, f'k{k}.d_auroc_near')} | {f(q['d_auroc_far'])}{ci(c, f'k{k}.d_auroc_far')} | {f(q['curve']['r50'], 2)} | {f(q['curve']['r10'], 2)} | {f(q['d_r10'], 2)}{ci(c, f'k{k}.d_r10', 2)} |")
        print(f"| {j} | horizon slope [imag - real] near / far | {f(p.get('horizon_slope_near'), 4)}{ci(c, 'horizon_slope_near', 4)} / {f(p.get('horizon_slope_far'), 4)}{ci(c, 'horizon_slope_far', 4)} | | | | | | | | | | | | |")
    print()

print("## per task, k = 1 (verdict components)\n")
print("| cell | near | far | near/far | slope (fail) [CI] | dAUROC near [CI] | dAUROC far | horizon slope [CI] | H1a | H1b |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, c in cells.items():
    if name.endswith("pooled6"):
        continue
    p, v = c["point"], c["verdict"]
    q = p.get("k1", {})
    print(f"| {name} | {f(q.get('near'))} | {f(q.get('far'))} | {f(q.get('near_over_far'), 2)} | {f(q.get('slope_fail'), 2)}{ci(c, 'k1.slope_fail', 2)} | {f(q.get('d_auroc_near'))}{ci(c, 'k1.d_auroc_near')} | {f(q.get('d_auroc_far'))} | {f(p.get('horizon_slope_near'), 4)}{ci(c, 'horizon_slope_near', 4)} | {v['H1a']['holds']} | {v['H1b']['holds']} |")
print()

print("## binned curve P(S_hat = 1) by r bin, pooled6\n")
bins = d["bins"]
hdr = ["<0.8"] + [f"{b[0]:.1f}-{b[1]:.1f}" for b in bins[1:-1]] + [">=3"]
print("| judge | frames | " + " | ".join(hdr) + " |")
print("|---|---|" + "---|" * len(hdr))
for j in ("a", "c", "d"):
    c = cells.get(f"{j}|pooled6")
    if c is None:
        continue
    p = c["point"]
    print(f"| {j} | real | " + " | ".join(f(x, 2) for x in p["real"]["bins"]) + " |")
    for k in (1, 2, 3, 4):
        if f"k{k}" in p:
            print(f"| {j} | k={k} | " + " | ".join(f(x, 2) for x in p[f"k{k}"]["bins"]) + " |")
print()

print("## sensitivity (pooled6): slope of S_hat on log r over failures r >= 1\n")
print("| judge, k | task-intercept slope [CI] | GLMM (start-state RE) slope [Wald CI] | sigma |\n|---|---|---|---|")
for key, s in d.get("sensitivity_pooled6", {}).items():
    g = s["glmm"]
    print(f"| {key} | {f(s['task_intercept_slope'], 2)} [{f(s['task_intercept_slope_ci95'][0], 2)}, {f(s['task_intercept_slope_ci95'][1], 2)}] | {f(g['beta'], 2)} [{f(g['beta_ci95'][0], 2)}, {f(g['beta_ci95'][1], 2)}] | {f(g['sigma'], 2)} |")
print()

if "D3" in d:
    print("## D3 information ceiling (feature-trained discriminator, out-of-fold)\n")
    print("| features | group | AUROC s-vs-near | AUROC s-vs-far | AUROC s-vs-fail |\n|---|---|---|---|---|")
    for key, res in d["D3"].items():
        for g in ("pooled6", "t0"):
            if g in res:
                x = res[g]
                print(f"| {key} | {g} | {f(x['auroc_near'])} | {f(x['auroc_far'])} | {f(x['auroc_fail'])} |")
    print()
