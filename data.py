"""Download TinyStories, train the tokenizer and dump the token ids."""
import argparse
import os
import urllib.request

import numpy as np
from tokenizers import ByteLevelBPETokenizer

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
DATA = "data"
VOCAB = 4096
EOT = "<|endoftext|>"


def main():
    p = argparse.ArgumentParser()
    # ~180M tokens, enough for the 164M-token main run
    p.add_argument("--mb", type=int, default=700, help="MB of the 2.2 GB TinyStories train file to download")
    args = p.parse_args()
    os.makedirs(DATA, exist_ok=True)

    raw = f"{DATA}/tinystories_{args.mb}mb.txt"
    if not os.path.exists(raw):
        print(f"downloading the first {args.mb} MB of TinyStories...")
        # range request, only download what we need
        req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{args.mb * 2**20 - 1}"})
        with urllib.request.urlopen(req) as r, open(raw + ".part", "wb") as f:
            f.write(r.read())
        os.replace(raw + ".part", raw)  # don't cache a half-finished download
    text = open(raw, encoding="utf-8", errors="ignore").read()
    stories = [s.strip() for s in text.split(EOT)[:-1] if s.strip()]  # last story is cut off by the range
    print(f"{len(stories):,} stories")

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(stories, vocab_size=VOCAB, special_tokens=[EOT])
    tok.save(f"{DATA}/tokenizer.json")

    eot = tok.token_to_id(EOT)
    ids = np.concatenate([
        np.array([i for e in tok.encode_batch(stories[k:k + 10_000]) for i in e.ids + [eot]], dtype=np.uint16)
        for k in range(0, len(stories), 10_000)
    ])
    n = int(0.98 * len(ids))
    ids[:n].tofile(f"{DATA}/train.bin")
    ids[n:].tofile(f"{DATA}/val.bin")
    print(f"{len(ids):,} tokens ({len(text) / len(ids):.1f} chars/token) -> {DATA}/train.bin, {DATA}/val.bin")


if __name__ == "__main__":
    main()
