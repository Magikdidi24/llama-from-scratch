"""Sample a story from a checkpoint."""
import argparse

import torch
from tokenizers import Tokenizer

from data import DATA, EOT
from model import Llama, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="?", default="Once upon a time")
p.add_argument("--ckpt", default="out/ckpt.pt")
p.add_argument("--tokens", type=int, default=200)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=50)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.ckpt, map_location=device)
model = Llama(ModelConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
eot = tok.token_to_id(EOT)
idx = torch.tensor([tok.encode(args.prompt).ids or [eot]], device=device)  # an empty prompt starts a new story
# temperature 0 = greedy, can't divide the logits by 0
out = model.generate(idx, args.tokens, args.temperature or 1.0, 1 if args.temperature == 0 else args.top_k)[0].tolist()
print(tok.decode(out[:out.index(eot, idx.size(1))] if eot in out[idx.size(1):] else out))
