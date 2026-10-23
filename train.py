"""Pretrain the model on the tokens from data.py."""
import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from data import DATA, VOCAB
from model import Llama, ModelConfig


@dataclass
class TrainConfig:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    n_experts: int = 0  # 0 = dense
    top_k: int = 2
    aux_loss_coef: float = 0.01
    seq_len: int = 256
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 2000
    optimizer: str = "adamw"  # or "muon"
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    data_dir: str = DATA
    out: str = "out/ckpt.pt"  # metrics go next to it as .json
    seed: int = 0
    compile: bool = False  # torch.compile, CUDA only


def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.seq_len - 1, (cfg.batch_size,)).tolist()
    xy = torch.from_numpy(np.stack([data[i:i + cfg.seq_len + 1] for i in ix]).astype(np.int64)).to(device)
    return xy[:, :-1], xy[:, 1:]


def lr_at(step, cfg):
    # linear warmup, then cosine decay down to min_lr
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizers(model, cfg):
    if cfg.optimizer not in ("adamw", "muon"):
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    # muon only for the 2D weights in the blocks, not the routers
    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if cfg.optimizer == "muon" and name.startswith("layers.") and p.dim() == 2 and ".router." not in name:
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)  # norms
    opts = [torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))]
    if muon:
        # match_rms_adamw (moonshot) lets muon reuse adamw's lr and weight decay
        opts.append(torch.optim.Muon(muon, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw"))
    return opts


@torch.no_grad()
def evaluate(model, data, cfg, device, autocast):
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        with autocast:
            losses.append(model(*get_batch(data, cfg, device))[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    train_data = np.memmap(f"{cfg.data_dir}/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(f"{cfg.data_dir}/val.bin", dtype=np.uint16, mode="r")

    mcfg = ModelConfig(VOCAB, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.seq_len,
                       n_experts=cfg.n_experts, top_k=cfg.top_k, aux_loss_coef=cfg.aux_loss_coef)
    raw_model = Llama(mcfg).to(device)
    torch.manual_seed(cfg.seed)  # reseed so all runs see the same batches
    torch.compiler.reset()  # new model every run, otherwise the scaling sizes hit the recompile limit
    model = torch.compile(raw_model) if cfg.compile and device == "cuda" else raw_model
    params, active = raw_model.num_params(), raw_model.num_params(active=True)
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len
    print(f"{active / 1e6:.2f}M active / {params / 1e6:.2f}M total params (non-embedding) on {device}, "
          f"{tokens_per_step * cfg.max_steps / 1e6:.1f}M training tokens")

    opts = make_optimizers(raw_model, cfg)
    groups = [g for opt in opts for g in opt.param_groups]

    records, train_time, t0 = [], 0.0, time.perf_counter()
    for step in range(cfg.max_steps):
        for g in groups:
            g["lr"] = lr_at(step, cfg)
        for _ in range(cfg.grad_accum):
            with autocast:
                _, loss, aux = model(*get_batch(train_data, cfg, device))
            ((loss + mcfg.aux_loss_coef * aux) / cfg.grad_accum).backward()
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)

        evaluating = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps
        if step % 50 == 0 or evaluating:
            r = {"step": step + 1, "tokens": (step + 1) * tokens_per_step, "train_loss": loss.item(),
                 "aux_loss": aux.item(), "lr": lr_at(step, cfg)}
            dt = time.perf_counter() - t0  # .item() already synced, eval time isn't counted
            train_time += dt
            # since the last record, so compile time only hits the first one
            r["tok_s"] = (r["tokens"] - (records[-1]["tokens"] if records else 0)) / dt
            print(f"step {r['step']:5d} | loss {r['train_loss']:.3f} | aux {r['aux_loss']:.3f} | lr {r['lr']:.1e} "
                  f"| grad norm {norm:.2f} | {r['tok_s']:,.0f} tok/s")
            if evaluating:
                r["val_loss"] = evaluate(model, val_data, cfg, device, autocast)
                print(f"step {r['step']:5d} | val loss {r['val_loss']:.3f}")
            records.append(r)
            t0 = time.perf_counter()

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    torch.save({"model": raw_model.state_dict(), "config": asdict(mcfg)}, cfg.out)
    metrics = {"config": asdict(cfg), "params": params, "active_params": active,
               "val_loss": records[-1]["val_loss"], "tok_s": records[-1]["tok_s"], "records": records}
    with open(os.path.splitext(cfg.out)[0] + ".json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"saved {cfg.out} and its metrics ({train_time / 60:.1f} min of training)")
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if isinstance(f.default, bool):
            p.add_argument(f"--{f.name}", action="store_true")
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())
