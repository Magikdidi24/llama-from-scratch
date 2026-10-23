import json
import math

import numpy as np
import pytest
import torch

from model import Llama, ModelConfig, apply_rope, rope_cache
from train import TrainConfig, get_batch, lr_at, make_optimizers, train


def tiny(**kw):
    torch.manual_seed(0)
    return Llama(ModelConfig(vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16, **kw))


def test_is_causal():
    m = tiny().eval()
    x = torch.randint(0, 50, (2, 16))
    y = x.clone()
    y[:, -1] = (x[:, -1] + 1) % 50  # change only the last token
    assert torch.allclose(m(x)[0][:, :-1], m(y)[0][:, :-1], atol=1e-5)


def test_overfits_one_batch():
    m = tiny()
    x = torch.randint(0, 50, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(150):
        loss = m(x[:, :-1], x[:, 1:])[1]
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.1, loss.item()


def test_rope_matches_complex_rotation():
    # reference: rotation as a complex multiplication
    T, d = 16, 8
    x = torch.randn(2, 4, T, d)
    angles = torch.outer(torch.arange(T).float(), 1 / 10000 ** (torch.arange(0, d, 2) / d))
    rot = torch.polar(torch.ones_like(angles), angles)
    ref = torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (-1, 2))) * rot).flatten(-2)
    assert torch.allclose(apply_rope(x, *rope_cache(d, T, 10000.0)), ref, atol=1e-6)


def test_get_batch_targets_are_next_tokens():
    x, y = get_batch(np.arange(1000, dtype=np.uint16), TrainConfig(seq_len=16, batch_size=4), "cpu")
    assert x.shape == (4, 16) and torch.equal(y, x + 1)


def test_lr_schedule():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert math.isclose(lr_at(9, cfg), 1.0)
    assert math.isclose(lr_at(60, cfg), 0.55)
    assert math.isclose(lr_at(110, cfg), 0.1)


@pytest.mark.parametrize("n_experts", [0, 4])
def test_kv_cache_matches_full_recompute(n_experts):
    m = tiny(n_experts=n_experts).eval()
    for p in m.parameters():  # default init is too small for attention to matter here
        torch.nn.init.normal_(p, std=0.3)
    prompt = torch.randint(0, 50, (2, 5))
    out = m.generate(prompt, max_new_tokens=100, top_k=1)  # greedy, stops at max_seq_len = 16
    ref = prompt
    while ref.size(1) < 16:
        ref = torch.cat([ref, m(ref)[0][:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref)


def test_moe_params_are_compute_matched():
    dense, moe = tiny(), tiny(n_experts=4)  # top_k = 2, so experts are half as wide
    router = 2 * 32 * 4
    ffn = sum(p.numel() for layer in dense.layers for p in layer.ffn.parameters())
    assert moe.num_params(active=True) == dense.num_params() + router
    assert moe.num_params() == dense.num_params() + router + ffn  # 4 half-width experts = 2x the ffn params


def test_moe_overfits_one_batch():
    m = tiny(n_experts=4)
    x = torch.randint(0, 50, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss, aux = m(x[:, :-1], x[:, 1:])
        (loss + m.config.aux_loss_coef * aux).backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.1, loss.item()


# uniform probs give 1, all tokens on 2 of 4 experts give n_experts / top_k = 2
@pytest.mark.parametrize("logits, expected", [([0., 0., 0., 0.], 1.0), ([9., 9., -9., -9.], 2.0)],
                         ids=["uniform_probs", "collapsed"])
def test_balancing_loss(logits, expected):
    m = tiny(n_experts=4)
    for layer in m.layers:
        layer.ffn.router.forward = lambda t: torch.tensor(logits).expand(len(t), 4)
    aux = m(torch.randint(0, 50, (2, 16)))[2]
    assert math.isclose(aux.item(), expected, rel_tol=1e-6)


def test_muon_step():
    m = tiny(n_experts=4)
    adamw, muon = make_optimizers(m, TrainConfig(optimizer="muon"))
    names = {p: n for n, p in m.named_parameters()}
    on_muon = {names[p] for p in muon.param_groups[0]["params"]}
    not_on_muon = {"tok_emb.weight", "layers.0.ffn.router.weight", "layers.1.ffn.router.weight"}
    assert on_muon == {n for n, p in m.named_parameters() if p.dim() == 2} - not_on_muon
    assert sum(len(g["params"]) for g in adamw.param_groups) + len(on_muon) == len(names)

    x = torch.randint(0, 50, (4, 17))
    before = [p.detach().clone() for p in m.parameters()]
    loss = m(x[:, :-1], x[:, 1:])[1]
    loss.backward()
    adamw.step()
    muon.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, m.parameters()))
    assert m(x[:, :-1], x[:, 1:])[1] < loss


@pytest.mark.parametrize("variant", [{}, {"n_experts": 4, "optimizer": "muon"}])
def test_train_end_to_end(tmp_path, monkeypatch, variant):
    opts = []  # to check that train() sets their lr

    def spy(*args):
        opts.extend(make_optimizers(*args))
        return opts

    monkeypatch.setattr("train.make_optimizers", spy)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 4096, 5000, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    cfg = TrainConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, seq_len=16, batch_size=4, max_steps=4,
                      warmup_steps=2, eval_every=2, eval_batches=2, data_dir=str(tmp_path),
                      out=str(tmp_path / "ckpt.pt"), **variant)
    result = train(cfg)
    assert math.isfinite(result["val_loss"])
    metrics = json.loads((tmp_path / "ckpt.json").read_text())
    assert [r["step"] for r in metrics["records"] if "val_loss" in r] == [2, 4]
    assert metrics["config"]["optimizer"] == cfg.optimizer
    assert {g["lr"] for opt in opts for g in opt.param_groups} == {lr_at(cfg.max_steps - 1, cfg)}
    ckpt =torch.load(tmp_path / "ckpt.pt")
    Llama(ModelConfig(**ckpt["config"])).load_state_dict(ckpt["model"])
