"""Visual-token selection rules given an importance vector p over the merged token grid (row-major)."""
import math, numpy as np

def grid_shape(grid_thw):
    t, hp, wp = grid_thw                     # patch grid; tokens are 2x2-merged
    return hp // 2, wp // 2

def _stratified(p, gh, gw, k, avail=None):
    """Partition the gh x gw token grid into ~k cells; keep the highest-p available token per cell."""
    n = gh * gw
    if avail is None: avail = np.ones(n, dtype=bool)
    k = max(1, min(k, int(avail.sum())))
    r = max(1, int(round(math.sqrt(k * gh / gw)))); c = max(1, int(math.ceil(k / r)))
    rows = np.minimum((np.arange(gh) * r) // gh, r - 1); cols = np.minimum((np.arange(gw) * c) // gw, c - 1)
    cell = (rows[:, None] * c + cols[None, :]).reshape(-1)
    keep = []
    for cid in np.unique(cell):
        idx = np.where((cell == cid) & avail)[0]
        if len(idx): keep.append(idx[np.argmax(p[idx])])
    keep = np.array(keep, dtype=np.int64)
    if len(keep) > k:                       # too many cells: keep the k most important of the cell winners
        keep = keep[np.argsort(-p[keep])[:k]]
    elif len(keep) < k:                     # too few: fill with next-best available tokens
        rest = np.setdiff1d(np.where(avail)[0], keep); rest = rest[np.argsort(-p[rest])][: k - len(keep)]
        keep = np.concatenate([keep, rest])
    return keep

def select(p, grid_thw, eta, mode="topk", H=None, rng=None):
    p = np.asarray(p, dtype=np.float64); n = len(p); gh, gw = grid_shape(grid_thw); assert gh * gw == n, (gh, gw, n)
    k = max(1, math.ceil(eta * n))
    if eta >= 1.0: return np.arange(n)
    if mode == "topk": return np.argsort(-p)[:k]
    if mode == "random": return (rng or np.random.default_rng(0)).choice(n, size=k, replace=False)
    if mode == "stratified": return _stratified(p, gh, gw, k)
    if mode in ("hybrid50", "adaptive"):
        frac_cov = 0.5 if mode == "hybrid50" else float(np.clip(H, 0.0, 1.0))
        k_cov = int(round(frac_cov * k)); k_top = k - k_cov
        top = np.argsort(-p)[:k_top]; avail = np.ones(n, dtype=bool); avail[top] = False
        cov = _stratified(p, gh, gw, k_cov, avail) if k_cov > 0 else np.array([], dtype=np.int64)
        return np.concatenate([top, cov])
    raise ValueError(mode)

if __name__ == "__main__":
    rng = np.random.default_rng(0); p = rng.random(13 * 20); p[50] = 5; p[51] = 4
    for m in ["topk", "random", "stratified", "hybrid50", "adaptive"]:
        s = select(p, (1, 26, 40), 0.1, m, H=0.77); print(m, len(s), sorted(s)[:8], "unique" if len(set(s)) == len(s) else "DUP")
