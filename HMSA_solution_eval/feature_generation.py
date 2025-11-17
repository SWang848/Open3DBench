import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Example hierarchy strings
paths = [
    "a__12__!",   # p0
    "a__12__@",   # p1
    "a__1__@",    # p2
    "b__3__!",    # p3
]

def split_to_tokens(path_str):
    return path_str.split("__")

def prefixes_from_tokens(tokens):
    # ['a','12','!'] -> ['a', 'a/12', 'a/12/!']
    out = []
    acc = []
    for t in tokens:
        acc.append(t)
        out.append("/".join(acc))
    return out

# Collect all prefixes
all_prefixes = OrderedDict()
for p in paths:
    tokens = split_to_tokens(p)
    prefs = prefixes_from_tokens(tokens)
    for pr in prefs:
        all_prefixes.setdefault(pr, None)

# Add PAD as id 0
PAD = 0
prefix2id = {"<PAD>": PAD}
for i, pr in enumerate(all_prefixes.keys(), start=1):
    prefix2id[pr] = i

id2prefix = {i: p for p, i in prefix2id.items()}

print("Prefix vocab:")
for k, v in prefix2id.items():
    print(v, "->", k)

max_depth = max(len(prefixes_from_tokens(split_to_tokens(p))) for p in paths)

def path_to_prefix_ids(path_str):
    tokens = split_to_tokens(path_str)
    prefs = prefixes_from_tokens(tokens)
    ids = [prefix2id[pr] for pr in prefs]
    # right-pad with PAD to max_depth
    while len(ids) < max_depth:
        ids.append(PAD)
    return ids

prefix_id_seqs = torch.tensor([path_to_prefix_ids(p) for p in paths], dtype=torch.long)
# shape: [B, L]  where B = len(paths), L = max_depth
print("prefix_id_seqs:")
print(prefix_id_seqs)

class FrozenPrefixSumEncoder(nn.Module):
    def __init__(self, num_prefixes, d=16, pad_id=0, alpha=1.0, max_depth=8, seed=123):
        super().__init__()
        self.pad_id = pad_id
        self.alpha = alpha

        # fixed random embeddings
        g = torch.Generator().manual_seed(seed)
        E = torch.randn(num_prefixes, d, generator=g) / (d ** 0.5)
        E[pad_id] = 0.0  # PAD -> zero vector

        self.emb = nn.Embedding.from_pretrained(E, freeze=True, padding_idx=pad_id)
        self.register_buffer("pos", torch.arange(max_depth).float())

    def forward(self, prefix_ids):  # [B, L]
        B, L = prefix_ids.shape
        x = self.emb(prefix_ids)  # [B, L, d]
        mask = (prefix_ids != self.pad_id).float()  # [B, L]

        if self.alpha != 1.0:
            # geometric depth weights
            w = (self.alpha ** self.pos[:L])[None, :, None]  # [1, L, 1]
            x = x * w
            denom = (w.squeeze(-1) * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)

        h = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, d]
        return h
    
num_prefixes = len(prefix2id)   # includes PAD
encoder = FrozenPrefixSumEncoder(
    num_prefixes=num_prefixes,
    d=64,
    pad_id=PAD,
    alpha=1.0,      # equal weights; change to 0.7 later if you want
    max_depth=max_depth,
    seed=123,
)

h = encoder(prefix_id_seqs)     # [B, d]
print("Embeddings shape:", h.shape)

def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

for i in range(len(paths)):
    for j in range(i+1, len(paths)):
        s = cos_sim(h[i], h[j])
        print(f"cos_sim({paths[i]} , {paths[j]}) = {s:.3f}")
