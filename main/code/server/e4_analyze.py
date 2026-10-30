"""B E4 analysis (DESIGN_main v3.1 section 9; CPU analysis env).
  4a  M1: S97 vs S50 fine-tuning (3 seeds pooled per record) on val H5 r (primary, judge a_full), val H5 r_obj, trainsplit
      H5 r: NM / SM, paired differences (cluster bootstrap over source trajectories, 10,000); each vs the public model.
      M3: p = 0.03 vs p = 0.3 (E1 grid, seeds pooled, scene bootstrap) and the dose-response with p = 0.1.
      H4a holds if M1 or M3 shows NM up and SM down, both with paired 95% CIs excluding 0.
  4b  H4b per model: real-frame ceiling (judge (a) on real frames; M3 frozen classifier) >= 0.95 and
      real - imagined ceiling (judge trained on imagined features) >= 0.05, AUROC success vs near (point estimates,
      paired bootstrap CI of the drop). M1 per E1 cell (primary val H5 r), M3 p = 0.03 seeds pooled.
      M3 resolution sweep R1-R4 (descriptive) next to the E1 base.
  4c  E4c JSONs (M1, M3).
Writes results/E4/e4_analysis.json. usage: python e4_analyze.py [B]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e45lib as L

R4 = os.path.expanduser('~/pmx/B/results/E4')


def wauc(sp, wp, sn, wn):
    if len(sp) == 0 or len(sn) == 0 or wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    o = np.argsort(sn)
    s, w = sn[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    lo, hi = np.searchsorted(s, sp, 'left'), np.searchsorted(s, sp, 'right')
    return float(np.sum(wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))) / (wp.sum() * wn.sum()))


def auc_pair(sr, si, band, unit, W, n_boot=1000):
    """AUROC success vs near for real and imagined-ceiling scores; drop = real - ceiling with bootstrap CI."""
    sm, nm = band == 'success', band == 'near'
    one = np.ones(len(sr))
    ar, ai = wauc(sr[sm], one[sm], sr[nm], one[nm]), wauc(si[sm], one[sm], si[nm], one[nm])
    d = []
    for b in range(min(n_boot, W.shape[0])):
        w = W[b][unit]
        d.append(wauc(sr[sm], w[sm], sr[nm], w[nm]) - wauc(si[sm], w[sm], si[nm], w[nm]))
    return dict(real=ar, imag_ceiling=ai, drop=ar - ai, ci_drop=L.ci(d))


def m1_4a(B):
    out = {}
    for tag in ('val_H5_r', 'val_H5_r_obj', 'trainsplit_H5_r'):
        if not os.path.exists(f'{L.E4M1}/imag_{tag}_S97_s2.npz') or not os.path.exists(f'{L.E4M1}/imag_{tag}_S50_s2.npz'):
            continue
        C = L.M1Cell(tag)
        W = L.boot_w(C.n_units, B, 4401)
        J = {v: C.judged(v) for v in ('orig', 'S97', 'S50')}
        res = dict(n_near=int((C.band == 'near').sum()), n_success=int((C.band == 'success').sum()),
                   orig_vs_e1_mismatches=int((J['orig'] != C.e1_imag).sum()),
                   perception_floor_NM=float(C.real[C.band == 'near'].mean()), perception_SM=float(1 - C.real[C.band == 'success'].mean()))
        for v, Jv in J.items():
            pt, bs = L.nm_sm(Jv, C.band, C.unit, W)
            res[v] = dict(NM=pt['NM'], SM=pt['SM'], ci_NM=L.ci(bs['NM']), ci_SM=L.ci(bs['SM']))
        res['S97_minus_S50'] = L.paired(J['S97'], J['S50'], C.band, C.unit, W)
        res['S97_minus_orig'] = L.paired(J['S97'], J['orig'], C.band, C.unit, W)
        res['S50_minus_orig'] = L.paired(J['S50'], J['orig'], C.band, C.unit, W)
        d = res['S97_minus_S50']
        res['H4a_holds'] = bool(d['NM']['diff'] > 0 and d['NM']['ci'][0] > 0 and d['SM']['diff'] < 0 and d['SM']['ci'][1] < 0)
        out[tag] = res
        print(f"M1 4a {tag}: NM orig/S97/S50 {res['orig']['NM']:.3f}/{res['S97']['NM']:.3f}/{res['S50']['NM']:.3f}, SM {res['orig']['SM']:.3f}/"
              f"{res['S97']['SM']:.3f}/{res['S50']['SM']:.3f} | S97-S50 NM {d['NM']['diff']:+.3f} {d['NM']['ci']} SM {d['SM']['diff']:+.3f} {d['SM']['ci']}"
              f" -> {res['H4a_holds']}", flush=True)
    return out


def m3_4a(B):
    G = L.M3Grid()
    W = L.boot_w(G.n_units, B, 4402)
    J = {p: G.judged_e1(p) for p in ('0.03', '0.1', '0.3')}
    res = dict(n_near=int((G.band == 'near').sum()), n_success=int((G.band == 'success').sum()))
    for p, Jp in J.items():
        pt, bs = L.nm_sm(Jp, G.band, G.unit, W)
        res[f'p{p}'] = dict(NM=pt['NM'], SM=pt['SM'], ci_NM=L.ci(bs['NM']), ci_SM=L.ci(bs['SM']))
    res['p0.03_minus_p0.3'] = L.paired(J['0.03'], J['0.3'], G.band, G.unit, W)
    res['p0.03_minus_p0.1'] = L.paired(J['0.03'], J['0.1'], G.band, G.unit, W)
    res['p0.1_minus_p0.3'] = L.paired(J['0.1'], J['0.3'], G.band, G.unit, W)
    d = res['p0.03_minus_p0.3']
    res['H4a_holds'] = bool(d['NM']['diff'] > 0 and d['NM']['ci'][0] > 0 and d['SM']['diff'] < 0 and d['SM']['ci'][1] < 0)
    print(f"M3 4a: NM p.03/.1/.3 {res['p0.03']['NM']:.3f}/{res['p0.1']['NM']:.3f}/{res['p0.3']['NM']:.3f}, SM {res['p0.03']['SM']:.4f}/"
          f"{res['p0.1']['SM']:.4f}/{res['p0.3']['SM']:.4f} | .03-.3 NM {d['NM']['diff']:+.3f} {d['NM']['ci']} SM {d['SM']['diff']:+.4f} {d['SM']['ci']}"
          f" -> {res['H4a_holds']}", flush=True)
    return res


def m1_4b(B):
    out = {}
    for H in (3, 5, 7, 10):
        for d in ('r', 'r_obj'):
            tag = f'val_H{H}_{d}'
            recs = [json.loads(l) for l in open(f'{L.E1M1}/eval_{tag}.jsonl')]
            cf = np.array([r['kind'] == 'cf' for r in recs])
            x = np.array([r[d] for r in recs])[cf]
            band = np.where(x < 1, 'success', np.where(x < 2, 'near', 'other'))
            tr = np.array([r['traj'] for r in recs])[cf]
            ut = np.unique(tr)
            unit = np.searchsorted(ut, tr)
            W = L.boot_w(len(ut), 1000, 4403)
            key = 'p_full' if d == 'r' else 'p_obj'
            ck = 'pi_full_imag' if d == 'r' else 'pi_obj_imag'
            sr = np.array([r[f'{key}_real'] for r in recs])[cf]
            si = np.array([r[ck] for r in recs])[cf]
            out[tag] = auc_pair(sr, si, band, unit, W)
    p = out['val_H5_r']
    out['H4b_M1'] = dict(cell='val_H5_r', real=p['real'], imag_ceiling=p['imag_ceiling'], drop=p['drop'], ci_drop=p['ci_drop'],
                         holds=bool(p['real'] >= 0.95 and p['drop'] >= 0.05))
    print(f"M1 4b val H5 r: real {p['real']:.3f} ceiling {p['imag_ceiling']:.3f} drop {p['drop']:.3f} {p['ci_drop']} -> {out['H4b_M1']['holds']}", flush=True)
    return out


def m3_4b():
    out = {}
    for p in ('0.03', '0.1', '0.3'):
        sr, si, band, unit = [], [], [], []
        for k in (0, 1, 2):
            z = np.load(f'{L.E1M3}/e1_p{p}_std_1x_s{k}.npz')
            g, s = np.where(z['use'])
            r = z['grid_r'][g]
            sr.append(z['real_logit'][g, s]); si.append(z['ceil_logit'][g, s])
            band.append(np.where(r < 1, 'success', np.where(r < 2, 'near', 'other'))); unit.append(s)
        sr, si, band, unit = (np.concatenate(v) for v in (sr, si, band, unit))
        W = L.boot_w(int(unit.max()) + 1, 1000, 4404)
        out[f'p{p}'] = auc_pair(sr, si, band, unit, W)
    q = out['p0.03']
    out['H4b_M3'] = dict(cell='p0.03', real=q['real'], imag_ceiling=q['imag_ceiling'], drop=q['drop'], ci_drop=q['ci_drop'],
                         holds=bool(q['real'] >= 0.95 and q['drop'] >= 0.05))
    print(f"M3 4b p=0.03: real {q['real']:.3f} ceiling {q['imag_ceiling']:.3f} drop {q['drop']:.3f} {q['ci_drop']} -> {out['H4b_M3']['holds']}", flush=True)
    return out


def sweep():
    out = {}
    one_auc = lambda a, b: wauc(a, np.ones(len(a)), b, np.ones(len(b)))
    variants = [('base', [f'{L.E1M3}/e1_p0.03_std_1x_s{k}.npz' for k in (0, 1, 2)])]
    variants += [(v, [f'{R4}/m3/sweep_{v}_s{k}.npz' for k in (0, 1, 2)]) for v in ('R1', 'R2', 'R3', 'R4')]
    for v, files in variants:
        if not all(os.path.exists(f) for f in files):
            continue
        sr, si, sc, band, rr = [], [], [], [], []
        for f in files:
            z = np.load(f)
            g, s = np.where(z['use'])
            r = z['grid_r'][g]
            sr.append(z['real_logit'][g, s]); si.append(z['imag_logit'][g, s]); sc.append(z['ceil_logit'][g, s])
            band.append(np.where(r < 1, 'success', np.where(r < 2, 'near', np.where(r < 3, 'mid', 'far')))); rr.append(r)
        sr, si, sc, band, rr = (np.concatenate(x) for x in (sr, si, sc, band, rr))
        sm, nm = band == 'success', band == 'near'
        ji = si > 0
        grid = np.unique(rr)
        curve = np.array([ji[rr == x].mean() for x in grid])
        r50 = next((float(grid[i - 1] + (curve[i - 1] - 0.5) / (curve[i - 1] - curve[i]) * (grid[i] - grid[i - 1]))
                    for i in range(1, len(grid)) if curve[i - 1] >= 0.5 > curve[i]), float('nan'))
        out[v] = dict(auc_sn_real=one_auc(sr[sm], sr[nm]), auc_sn_imag=one_auc(si[sm], si[nm]), auc_sn_ceiling=one_auc(sc[sm], sc[nm]),
                      NM=float(ji[nm].mean()), SM=float(1 - ji[sm].mean()), real_NM=float((sr[nm] > 0).mean()), r50_imag=r50)
        print(f"sweep {v}: AUROC real/imag/ceil {out[v]['auc_sn_real']:.3f}/{out[v]['auc_sn_imag']:.3f}/{out[v]['auc_sn_ceiling']:.3f} "
              f"NM {out[v]['NM']:.3f} SM {out[v]['SM']:.4f} r50 {r50:.3f}", flush=True)
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else L.B_BOOT
    t0 = time.time()
    out = dict(B=B)
    out['4a_M1'] = m1_4a(B)
    out['4a_M3'] = m3_4a(B)
    out['4b_M1'] = m1_4b(B)
    out['4b_M3'] = m3_4b()
    out['4b_sweep'] = sweep()
    for k, f in (('4c_M1', f'{R4}/m1/e4c_m1.json'), ('4c_M3', f'{R4}/m3/e4c_m3.json')):
        if os.path.exists(f):
            out[k] = json.load(open(f))
    m1 = out['4a_M1'].get('val_H5_r', {}).get('H4a_holds')
    out['verdict'] = dict(H4a_M1=m1, H4a_M3=out['4a_M3']['H4a_holds'], H4a=bool(m1 or out['4a_M3']['H4a_holds']),
                          H4b_M1=out['4b_M1']['H4b_M1']['holds'], H4b_M3=out['4b_M3']['H4b_M3']['holds'])
    out['time_s'] = time.time() - t0
    json.dump(out, open(f'{R4}/e4_analysis.json', 'w'), indent=1)
    print('VERDICT E4', json.dumps(out['verdict']), flush=True)
    print('E4_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
