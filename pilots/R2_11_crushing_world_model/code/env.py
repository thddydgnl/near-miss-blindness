"""R2-11 2D pushing domain: the R2-2 env (velocity-commanded agent disc with momentum pushing a square box inside
walls) extended with box ROTATION, so that box angle is a physical variable.

Changes vs R2-2 (code/env.py of R2_2_pretrained_twin_starvation):
  - box has angle th (rad) and angular velocity wb (rad/substep); square of half-size HB, inertia IB = MB*(2HB)^2/6
  - disc-vs-oriented-square contact with an inelastic normal impulse applied at the contact point (gives torque)
  - box-vs-wall clamp uses the rotated square's axis-aligned extent; a wall hit halves the angular velocity
  - renderer draws the rotated square
Everything else (constants, policy, substeps) is identical to R2-2.
State per env: pa, va, pb, vb ([...,2]), th, wb ([...]). The state is just these 10 numbers, so it can be saved and
restored exactly (CEM executes plans from saved states; counterfactuals re-simulate).
"""
import math

import torch

RES = 112          # render resolution (8x8 DINOv2 ViT-S/14 patches)
WALL = 4.0         # wall thickness (px)
RA = 8.0           # agent radius (px)
HB = 12.0          # box half-size (px) -> 24 px square
VMAX = 2.5         # agent speed command scale (px / substep at |a|=1 per dim)
KAPPA = 0.2        # agent velocity relaxation toward the command per substep (momentum)
MU = 0.12          # box linear velocity decay per substep (floor friction)
MU_W = 0.15        # box angular velocity decay per substep (floor friction, rotation)
MA, MB = 1.0, 1.5  # masses (agent, box)
IB = MB * (2 * HB) ** 2 / 6.0   # moment of inertia of the square box (=144)
K_SUB = 4          # physics substeps per control step (action held)
RHO = 0.85         # OU autocorrelation per control step
SIG_U = 0.55       # OU stationary std per dim
C_ATT = 0.35       # attraction strength toward (box + per-episode offset)
P_SWITCH = 0.1     # per-step probability that the attraction toggles on/off
OFF_STD = 5.0      # per-episode target offset std (px)
LO_A, HI_A = WALL + RA, RES - WALL - RA


def box_ext(th):
    """Half-width of the axis-aligned bounding box of the rotated square."""
    return HB * (torch.cos(th).abs() + torch.sin(th).abs())


def _rot(v, c, s):
    """Rotate 2-vectors v [...,2] by the angle with cosine c and sine s ([...])."""
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _clamp_walls(p, v, lo, hi):
    hit = (p > hi) | (p < lo)
    return p.clamp(lo, hi), torch.where(hit, torch.zeros_like(v), v)


def _clamp_box(pb, vb, th, wb):
    ext = box_ext(th)
    lo = (WALL + ext)[..., None]
    hi = (RES - WALL - ext)[..., None]
    hit = (pb > hi) | (pb < lo)
    pb = torch.maximum(torch.minimum(pb, hi), lo)
    vb = torch.where(hit, torch.zeros_like(vb), vb)
    wb = torch.where(hit.any(-1), 0.5 * wb, wb)
    return pb, vb, wb


def box_gap(pa, pb, th):
    """Signed gap (px) between the agent disc and the box surface (<=0 means touching/overlapping)."""
    c, s = torch.cos(th), torch.sin(th)
    d = _rot(pa - pb, c, -s)
    q = d.clamp(-HB, HB)
    out = (d - q).norm(dim=-1)
    inside = out < 1e-6
    pen_in = (HB - d.abs()).min(-1).values
    dist = torch.where(inside, -pen_in, out)
    return dist - RA


def wall_gap_box(pb, th):
    """Distance (px) from the rotated box's bounding extent to the nearest wall (0 = touching)."""
    ext = box_ext(th)[..., None]
    lo = pb - ext - WALL
    hi = (RES - WALL) - (pb + ext)
    return torch.minimum(lo, hi).min(-1).values.clamp_min(0.0)


def _contact(pa, va, pb, vb, th, wb, movable_box):
    """Disc (agent) vs oriented square (box): positional projection + perfectly inelastic normal impulse at the
    contact point (linear + angular). With movable_box=False the box is treated as pinned (only the agent moves)."""
    c, s = torch.cos(th), torch.sin(th)
    d = _rot(pa - pb, c, -s)                                # agent centre in the box frame
    q = d.clamp(-HB, HB)
    n = d - q
    dist = n.norm(dim=-1)
    inside = dist < 1e-6
    pen_ax = HB - d.abs()
    use_x = pen_ax[..., 0] < pen_ax[..., 1]
    zero = torch.zeros_like(dist)
    one = torch.ones_like(dist)
    sx = torch.where(d[..., 0] >= 0, one, -one)
    sy = torch.where(d[..., 1] >= 0, one, -one)
    n_in = torch.where(use_x[..., None], torch.stack([sx, zero], -1), torch.stack([zero, sy], -1))
    q_in = torch.where(use_x[..., None], torch.stack([sx * HB, d[..., 1]], -1), torch.stack([d[..., 0], sy * HB], -1))
    nl = torch.where(inside[..., None], n_in, n / dist.clamp_min(1e-6)[..., None])
    ql = torch.where(inside[..., None], q_in, q)
    pen = torch.where(inside, RA + pen_ax.min(-1).values, RA - dist)
    contact = pen > 0
    pen = pen.clamp_min(0.0)
    nrm = _rot(nl, c, s)                                     # contact normal (box -> agent), world frame
    r = _rot(ql, c, s)                                       # lever arm: contact point relative to box centre
    vcp = vb + wb[..., None] * torch.stack([-r[..., 1], r[..., 0]], -1)
    vn = ((va - vcp) * nrm).sum(-1)
    if movable_box:
        wa, wbm = MB / (MA + MB), MA / (MA + MB)
        pa = pa + nrm * (pen * wa)[..., None]
        pb = pb - nrm * (pen * wbm)[..., None]
        rn = r[..., 0] * nrm[..., 1] - r[..., 1] * nrm[..., 0]
        k = 1.0 / MA + 1.0 / MB + rn * rn / IB
        J = torch.where(contact & (vn < 0), -vn / k, zero)
        va = va + nrm * (J / MA)[..., None]
        vb = vb - nrm * (J / MB)[..., None]
        wb = wb - J * rn / IB
    else:
        pa = pa + nrm * pen[..., None]
        va = va - nrm * torch.where(contact & (vn < 0), vn, zero)[..., None]
    return pa, va, pb, vb, wb, contact


def control_step(pa, va, pb, vb, th, wb, a):
    """One control step = K_SUB substeps with the action held. Returns the new state and a contact flag."""
    touched = torch.zeros(pa.shape[:-1], dtype=torch.bool, device=pa.device)
    for _ in range(K_SUB):
        va = va + KAPPA * (VMAX * a - va)
        pa = pa + va
        vb = vb * (1.0 - MU)
        wb = wb * (1.0 - MU_W)
        pb = pb + vb
        th = th + wb
        pa, va, pb, vb, wb, c1 = _contact(pa, va, pb, vb, th, wb, True)
        pb, vb, wb = _clamp_box(pb, vb, th, wb)
        pa, va = _clamp_walls(pa, va, LO_A, HI_A)
        pa, va, _, _, _, c2 = _contact(pa, va, pb, vb, th, wb, False)   # box may be pinned: push the agent only
        pa, va = _clamp_walls(pa, va, LO_A, HI_A)
        touched = touched | c1 | c2
    return pa, va, pb, vb, th, wb, touched


def init_state(n, g, device):
    th = (math.pi / 2) * torch.rand(n, generator=g, device=device)
    ext = box_ext(th)[:, None]
    lo, hi = WALL + ext, RES - WALL - ext
    pb = lo + (hi - lo) * torch.rand(n, 2, generator=g, device=device)
    pa = LO_A + (HI_A - LO_A) * torch.rand(n, 2, generator=g, device=device)
    for _ in range(100):
        bad = (pa - pb).norm(dim=-1) < HB * math.sqrt(2) + RA + 2.0
        if not bad.any():
            break
        k = int(bad.sum())
        pa[bad] = LO_A + (HI_A - LO_A) * torch.rand(k, 2, generator=g, device=device)
    z = torch.zeros(n, 2, device=device)
    return pa, z.clone(), pb, z.clone(), th, torch.zeros(n, device=device)


def att_dir(pa, pb, off):
    t = pb + off - pa
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1.0)


def ou_action(u_prev, on, pa, pb, off, eps):
    """Teleop-like policy. u is the latent OU command; the attraction toggles via `on`."""
    u = RHO * u_prev + SIG_U * math.sqrt(1.0 - RHO * RHO) * eps
    a = (u + C_ATT * att_dir(pa, pb, off) * on[..., None].float()).clamp(-1.0, 1.0)
    return a, u


def simulate(n, T, policy, seed, device, burn=8):
    """Simulate n episodes; record T+1 states (frames 0..T) and T actions (a_t moves frame t -> t+1).
    Returns dict of tensors: pa,va,pb,vb [n,T+1,2]; th,wb [n,T+1]; a,u,u_prev [n,T,2]; on,contact [n,T]; off [n,2]."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pa, va, pb, vb, th, wb = init_state(n, g, device)
    u = SIG_U * torch.randn(n, 2, generator=g, device=device)
    on = torch.rand(n, generator=g, device=device) < 0.5
    off = OFF_STD * torch.randn(n, 2, generator=g, device=device)
    rec = {k: [] for k in ('pa', 'va', 'pb', 'vb', 'th', 'wb', 'a', 'u', 'u_prev', 'on', 'contact')}
    for t in range(burn + T + 1):
        r = t - burn
        if r >= 0:
            for k, v in (('pa', pa), ('va', va), ('pb', pb), ('vb', vb), ('th', th), ('wb', wb)):
                rec[k].append(v)
        if r == T:
            break
        flip = torch.rand(n, generator=g, device=device) < P_SWITCH
        on = on ^ flip
        eps = torch.randn(n, 2, generator=g, device=device)
        u_prev = u
        if policy == 'ou':
            a, u = ou_action(u, on, pa, pb, off, eps)
        elif policy == 'rand':
            a = torch.rand(n, 2, generator=g, device=device) * 2.0 - 1.0
        else:
            raise ValueError(policy)
        pa, va, pb, vb, th, wb, c = control_step(pa, va, pb, vb, th, wb, a)
        if r >= 0:
            rec['a'].append(a); rec['u'].append(u); rec['u_prev'].append(u_prev)
            rec['on'].append(on); rec['contact'].append(c)
    out = {k: torch.stack(v, 1) for k, v in rec.items()}
    out['off'] = off
    return out


# ---------------------------------------------------------------- rendering
BG = (0.92, 0.92, 0.90)
WALL_C = (0.30, 0.30, 0.32)
BOX_C = (0.85, 0.38, 0.12)
AGENT_C = (0.15, 0.32, 0.85)


def render(pa, pb, th, res=RES):
    """Anti-aliased RGB render (analytic 1-px soft edges). pa, pb: [N,2] (x, y) px; th: [N] rad.
    Returns [N,3,res,res] in [0,1]."""
    dev = pa.device
    N = pa.shape[0]
    c = torch.arange(res, device=dev, dtype=torch.float32) + 0.5
    ys, xs = torch.meshgrid(c, c, indexing='ij')
    img = torch.empty(N, 3, res, res, device=dev)
    for k in range(3):
        img[:, k] = BG[k]
    wall = ((xs < WALL) | (xs > res - WALL) | (ys < WALL) | (ys > res - WALL)).float()
    for k in range(3):
        img[:, k] = img[:, k] * (1 - wall) + WALL_C[k] * wall
    bx, by = pb[:, 0, None, None], pb[:, 1, None, None]
    cb, sb = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    dx, dy = xs - bx, ys - by
    lx, ly = cb * dx + sb * dy, -sb * dx + cb * dy          # pixel offsets in the box frame
    sd_b = torch.maximum(lx.abs(), ly.abs()) - HB
    cov_b = (0.5 - sd_b).clamp(0, 1)
    ax, ay = pa[:, 0, None, None], pa[:, 1, None, None]
    sd_a = torch.sqrt((xs - ax) ** 2 + (ys - ay) ** 2) - RA
    cov_a = (0.5 - sd_a).clamp(0, 1)
    for k in range(3):
        img[:, k] = img[:, k] * (1 - cov_b) + BOX_C[k] * cov_b
        img[:, k] = img[:, k] * (1 - cov_a) + AGENT_C[k] * cov_a
    return img
