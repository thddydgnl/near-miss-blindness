"""Shared helpers for using Cosmos Policy (LIBERO checkpoint) as an action-conditioned world model.

Runs inside the cosmos-policy uv venv, with the working directory at the cosmos-policy checkout
(the config file path in the policy config is relative).

Counterfactual query (same method as CureWM scripts/probe_optimism.py):
  1) build the observation batch and run the policy's action pass (this yields the clean latent sequence);
  2) overwrite the action latent frame with the given (normalized) 16-step action chunk
     using the model's own injection function replace_latent_with_action_chunk;
  3) get_future_state_prediction (1 denoising step, action frame as condition) -> imagined future
     third-person / wrist images and future proprio;
  4) get_value_prediction (1 denoising step) -> V(imagined future state) in [0, 1].
Extensions used in B:
  - imagine_chain: k chunks autoregressively (the imagined future images/proprio become the next observation);
  - value_real_future: the same value head applied to the *real* future state (real future frames encoded into
    the future latent slots, real future proprio injected), i.e. judge (c) on real frames.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch

from cosmos_policy.experiments.robot import cosmos_utils as CU
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_policy.utils.utils import duplicate_array

CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
# LIBERO latent layout: 0 blank, 1 curr proprio, 2 curr wrist, 3 curr primary, 4 action,
# 5 future proprio, 6 future wrist, 7 future primary, 8 value
IDX = dict(cur_proprio=1, cur_wrist=2, cur_primary=3, action=4, fut_proprio=5, fut_wrist=6, fut_primary=7, value=8)
PROPRIO_DIM = 9


def make_cfg(kind: str = "eval", **over) -> PolicyEvalConfig:
    """kind='eval': README quick start / LIBERO.md eval settings. kind='curewm': CureWM probe_optimism.make_cfg."""
    c = dict(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=CKPT,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{CKPT}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{CKPT}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True, unnormalize_actions=True,
        trained_with_image_aug=True, chunk_size=16, num_open_loop_steps=16,
        num_denoising_steps_action=5, num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        ar_future_prediction=False, ar_value_prediction=False,
        deterministic=True, use_variance_scale=False, randomize_seed=False, seed=195,
    )
    if kind == "eval":
        c.update(use_jpeg_compression=True, flip_images=True)
    elif kind == "curewm":
        c.update(use_jpeg_compression=False, flip_images=False, task_suite_name="libero_goal")
    else:
        raise ValueError(kind)
    c.update(over)
    return PolicyEvalConfig(**c)


def load_cosmos(cfg):
    stats = CU.load_dataset_stats(CU.resolve_path(cfg.dataset_stats_path))
    CU.init_t5_text_embeddings_cache(CU.resolve_path(cfg.t5_text_embeddings_path))
    model, _ = CU.get_model(cfg)
    return model, stats


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, key):
        sync()
        t0 = time.time()
        yield
        sync()
        self.t[key] = self.t.get(key, 0.0) + time.time() - t0


def env_obs_to_policy_obs(obs: dict, flip: bool = True) -> dict:
    """Same as run_libero_eval.prepare_observation (vertical flip, proprio = gripper2 + eef_pos3 + eef_quat4)."""
    img, wrist = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
    if flip:
        img, wrist = np.flipud(img), np.flipud(wrist)
    return {
        "primary_image": np.ascontiguousarray(img),
        "wrist_image": np.ascontiguousarray(wrist),
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }


def normalize_actions(a: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of cosmos_utils.unnormalize_actions (as in CureWM probe_optimism.normalize_actions)."""
    lo, hi = np.asarray(stats["actions_min"]), np.asarray(stats["actions_max"])
    return np.clip(2.0 * (a - lo) / np.maximum(hi - lo, 1e-8) - 1.0, -1.0, 1.0)


def unnormalize_proprio(p: np.ndarray, stats: dict) -> np.ndarray:
    lo, hi = np.asarray(stats["proprio_min"]), np.asarray(stats["proprio_max"])
    return (p + 1.0) / 2.0 * (hi - lo) + lo


def extract_proprio(latent: torch.Tensor, idx: int) -> np.ndarray:
    """Average the replicated proprio values in one latent frame (mirror of extract_action_chunk_from_latent_sequence)."""
    flat = latent[0, :, idx].reshape(-1).float()
    n = flat.shape[0] // PROPRIO_DIM
    return flat[: n * PROPRIO_DIM].reshape(n, PROPRIO_DIM).mean(0).cpu().numpy()


def build_batch(cfg, model, stats, obs: dict, lang, future_obs: dict | None = None, preprocessed: bool = False) -> dict:
    """LIBERO-only copy of the data_batch construction in cosmos_utils.get_action.

    future_obs=None reproduces get_action exactly (future slots hold copies of the current images).
    future_obs given: the future wrist/primary slots hold the real future images (for value_real_future).
    preprocessed=True: images are already in model space (224 px, test-time crop applied), e.g. imagined frames.
    """
    text_embedding = CU.get_t5_embedding_from_cache(lang) if isinstance(lang, str) else lang
    imgs = [obs["wrist_image"], obs["primary_image"]]
    if future_obs is not None:
        imgs += [future_obs["wrist_image"], future_obs["primary_image"]]
    proc = np.stack(imgs, axis=0) if preprocessed else CU.prepare_images_for_model(imgs, cfg)
    wrist, primary = proc[0], proc[1]
    fwrist, fprimary = (proc[2], proc[3]) if future_obs is not None else (wrist, primary)
    proprio = obs["proprio"]
    if cfg.normalize_proprio:
        proprio = CU.rescale_proprio(proprio, stats, non_negative_only=False, scale_multiplier=1.0)
    F4 = CU.COSMOS_TEMPORAL_COMPRESSION_FACTOR
    blank = np.zeros_like(primary)
    blank_d = duplicate_array(blank.copy(), total_num_copies=F4)
    seq = [np.expand_dims(np.zeros_like(blank), 0), blank_d,
           duplicate_array(wrist, total_num_copies=F4), duplicate_array(primary, total_num_copies=F4),
           blank_d.copy(), blank_d.copy(),
           duplicate_array(fwrist, total_num_copies=F4), duplicate_array(fprimary, total_num_copies=F4),
           blank_d.copy()]
    raw = np.concatenate(seq, axis=0)[None]
    raw = torch.from_numpy(np.transpose(raw, (0, 4, 1, 2, 3))).to(dtype=torch.uint8).cuda()
    t = lambda i: torch.tensor([i], dtype=torch.int64).cuda()
    return {
        "dataset_name": "video_data",
        "video": raw,
        "t5_text_embeddings": text_embedding.repeat(1, 1, 1).to(dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([16], dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((1, 1, CU.COSMOS_IMAGE_SIZE, CU.COSMOS_IMAGE_SIZE), dtype=torch.bfloat16).cuda(),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": torch.from_numpy(np.asarray(proprio)).reshape(1, -1).to(dtype=torch.bfloat16).cuda(),
        "current_proprio_latent_idx": t(IDX["cur_proprio"]),
        "current_wrist_image_latent_idx": t(IDX["cur_wrist"]),
        "current_wrist_image2_latent_idx": t(-1),
        "current_image_latent_idx": t(IDX["cur_primary"]),
        "current_image2_latent_idx": t(-1),
        "action_latent_idx": t(IDX["action"]),
        "future_proprio_latent_idx": t(IDX["fut_proprio"]),
        "future_wrist_image_latent_idx": t(IDX["fut_wrist"]),
        "future_wrist_image2_latent_idx": t(-1),
        "future_image_latent_idx": t(IDX["fut_primary"]),
        "future_image2_latent_idx": t(-1),
        "value_latent_idx": t(IDX["value"]),
    }


def _future_and_value(cfg, model, db, latent_with_action, orig, seed):
    fut = CU.get_future_state_prediction(
        cfg, model, data_batch=db, generated_latent_with_action=latent_with_action, orig_clean_latent_frames=orig,
        future_proprio_latent_idx=IDX["fut_proprio"], future_wrist_image_latent_idx=IDX["fut_wrist"],
        future_wrist_image2_latent_idx=-1, future_image_latent_idx=IDX["fut_primary"], future_image2_latent_idx=-1,
        seed=seed, num_denoising_steps_future_state=cfg.num_denoising_steps_future_state)
    val = CU.get_value_prediction(
        cfg, model, data_batch=db, future_state_samples_list=fut["future_state_samples_list"],
        seed=seed, num_denoising_steps_value=cfg.num_denoising_steps_value)
    return fut, val


@torch.inference_mode()
def cf_query(cfg, model, stats, obs, lang, chunk_raw, seed: int, n_act_steps: int | None = None,
             preprocessed: bool = False, timer: Timer | None = None) -> dict:
    """One counterfactual query (CureWM method). chunk_raw: (16, 7) LIBERO action units (gripper +1 close / -1 open)."""
    tm = timer or Timer()
    with tm("encode+action_pass"):
        db = build_batch(cfg, model, stats, obs, lang, preprocessed=preprocessed)
        gen, orig = model.generate_samples_from_batch(
            db, n_sample=1, num_steps=n_act_steps or cfg.num_denoising_steps_action, seed=seed,
            is_negative_prompt=False, use_variance_scale=cfg.use_variance_scale, return_orig_clean_latent_frames=True)
    policy_chunk = CU.extract_action_chunk_from_latent_sequence(
        gen, action_shape=(cfg.chunk_size, 7), action_indices=db["action_latent_idx"]).float().cpu().numpy()[0]
    a = torch.tensor(normalize_actions(np.asarray(chunk_raw, dtype=np.float64), stats)[None], dtype=gen.dtype, device=gen.device)
    inj = replace_latent_with_action_chunk(gen.clone(), a, db["action_latent_idx"])
    with tm("future+value"):
        fut, val = _future_and_value(cfg, model, db, inj, orig, seed)
    fs = fut["future_state_samples_list"][0]
    pred = fut["future_image_predictions"]
    return {
        "value": float(val["value_prediction"]),
        "future_image": np.asarray(pred["future_image"]),
        "future_wrist_image": np.asarray(pred["future_wrist_image"]),
        "future_proprio": unnormalize_proprio(extract_proprio(fs, IDX["fut_proprio"]), stats),
        "policy_chunk": CU.unnormalize_actions(policy_chunk[None], stats)[0],
        "times": dict(tm.t),
    }


def stack_batches(dbs: list) -> dict:
    """Concatenate single-item batches from build_batch along the batch dimension."""
    out = {}
    for k, v in dbs[0].items():
        if isinstance(v, torch.Tensor):
            out[k] = torch.cat([d[k] for d in dbs], dim=0)
        else:
            out[k] = v
    return out


@torch.inference_mode()
def cf_query_batch(cfg, model, stats, items: list, seed: int, preprocessed: bool = False, decode: bool = True) -> list:
    """Batched counterfactual queries. items: list of (obs, lang, chunk_raw). One action pass (1 step; equivalent,
    see quickstart.py), one future-state pass and one value pass for the whole batch."""
    B = len(items)
    db = stack_batches([build_batch(cfg, model, stats, o, l, preprocessed=preprocessed) for o, l, _ in items])
    gen, orig = model.generate_samples_from_batch(db, n_sample=B, num_steps=1, seed=seed, is_negative_prompt=False,
                                                  use_variance_scale=cfg.use_variance_scale,
                                                  return_orig_clean_latent_frames=True)
    a = torch.tensor(np.stack([normalize_actions(np.asarray(c, dtype=np.float64), stats) for _, _, c in items]),
                     dtype=gen.dtype, device=gen.device)
    inj = replace_latent_with_action_chunk(gen.clone(), a, db["action_latent_idx"])
    db["num_conditional_frames"] = model.config.min_num_conditional_frames + 1
    fs = model.generate_samples_from_batch(db, n_sample=B, num_steps=cfg.num_denoising_steps_future_state, seed=seed,
                                           is_negative_prompt=False, use_variance_scale=cfg.use_variance_scale,
                                           skip_vae_encoding=True, previous_generated_latent=inj)
    db["num_conditional_frames"] = model.config.state_t - 1
    db["mask_current_state_action_for_value_prediction"] = cfg.mask_current_state_action_for_value_prediction
    vl = model.generate_samples_from_batch(db, n_sample=B, num_steps=cfg.num_denoising_steps_value, seed=seed,
                                           is_negative_prompt=False, use_variance_scale=cfg.use_variance_scale,
                                           skip_vae_encoding=True, previous_generated_latent=fs)
    vals = CU.extract_value_from_latent_sequence(vl, torch.full((B,), -1, dtype=torch.int64, device=vl.device))
    vals = torch.clamp((vals + 1) / 2, 0, 1).float().cpu().numpy()
    imgs = None
    if decode:
        imgs = CU.get_future_images_from_generated_samples(
            model, fs.clone(), cfg, orig, [0, 1, 4, 5], IDX["fut_wrist"], -1, IDX["fut_primary"], -1,
            temporal_compression_factor=CU.COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    res = []
    for i in range(B):
        r = {"value": float(vals[i]),
             "future_proprio": unnormalize_proprio(extract_proprio(fs[i:i + 1], IDX["fut_proprio"]), stats)}
        if imgs is not None:
            r["future_image"] = imgs["future_image"][i]
            r["future_wrist_image"] = imgs["future_wrist_image"][i]
        res.append(r)
    return res


@torch.inference_mode()
def imagine_chain(cfg, model, stats, obs0, lang, chunks, seed: int, n_act_steps: int | None = None,
                  pre_first: bool = False) -> list:
    """Imagine len(chunks) chunks autoregressively from a real observation. Returns one cf_query result per chunk.
    pre_first=True: obs0 images are already in the model view (e.g. rendered + prepare_images_for_model)."""
    outs, obs, pre = [], obs0, pre_first
    for ch in chunks:
        q = cf_query(cfg, model, stats, obs, lang, ch, seed, n_act_steps=n_act_steps, preprocessed=pre)
        outs.append(q)
        obs = {"primary_image": q["future_image"], "wrist_image": q["future_wrist_image"], "proprio": q["future_proprio"]}
        pre = True
    return outs


@torch.inference_mode()
def value_real_future(cfg, model, stats, obs, lang, chunk_raw, future_obs, seed: int, preprocessed: bool = False) -> float:
    """Value head on the REAL future state: the real future frames are VAE-encoded into the future slots and the
    real future proprio is injected; the value frame is then generated exactly as in get_value_prediction."""
    db = build_batch(cfg, model, stats, obs, lang, future_obs=future_obs, preprocessed=preprocessed)
    gen, orig = model.generate_samples_from_batch(
        db, n_sample=1, num_steps=1, seed=seed, is_negative_prompt=False,
        use_variance_scale=cfg.use_variance_scale, return_orig_clean_latent_frames=True)
    lat = gen.clone()
    lat[:, :, IDX["fut_wrist"]:IDX["fut_primary"] + 1] = orig[:, :, IDX["fut_wrist"]:IDX["fut_primary"] + 1]
    a = torch.tensor(normalize_actions(np.asarray(chunk_raw, dtype=np.float64), stats)[None], dtype=lat.dtype, device=lat.device)
    lat = replace_latent_with_action_chunk(lat, a, db["action_latent_idx"])
    fp = CU.rescale_proprio(np.asarray(future_obs["proprio"]), stats, non_negative_only=False, scale_multiplier=1.0)
    lat = replace_latent_with_proprio(lat, torch.from_numpy(np.asarray(fp)).reshape(1, -1).to(lat.dtype).cuda(),
                                      proprio_indices=db["future_proprio_latent_idx"])
    val = CU.get_value_prediction(cfg, model, data_batch=db, future_state_samples_list=[lat], seed=seed,
                                  num_denoising_steps_value=cfg.num_denoising_steps_value)
    return float(val["value_prediction"])
