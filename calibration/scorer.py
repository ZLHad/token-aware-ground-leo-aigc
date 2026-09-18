"""User-side scoring pass on Qwen3-VL-2B (HF transformers, MPS).

For an (image, question): run the small model, take layer-K attention from the query
tokens (everything after <|vision_end|>) to the visual tokens, average over heads and
query positions, renormalize -> importance p_i over visual tokens. H = normalized entropy.
Also times (a) the truncated scoring pass (vision encoder + layers 0..K) and (b) the
full prefill, to estimate the FLOPs/latency ratio r = zeta_sc / zeta_pf_loc.
"""
import time, math, numpy as np, torch
from PIL import Image
from imgprep import canonical
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MIN_PIX, MAX_PIX = 256 * 32 * 32, 1024 * 32 * 32   # -> n_vis in [256, 1024] (one token = 32x32 px)
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

class _Stop(Exception): pass

class Scorer:
    def __init__(self, layer_k=2, device="mps", dtype=torch.float16):
        self.k = layer_k; self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIX, max_pixels=MAX_PIX)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="eager").to(device).eval()
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.vision_end_id = getattr(cfg, "vision_end_token_id", None) or self.proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._layers = self.model.model.language_model.layers if hasattr(self.model.model, "language_model") else self.model.model.layers
        self._hook = None

    def build_inputs(self, image: Image.Image, question: str):
        image = canonical(image)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + PROMPT_SUFFIX}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.proc(text=[text], images=[image], return_tensors="pt")

    @torch.no_grad()
    def score(self, image: Image.Image, question: str, time_it=True, layers=(2, 6, 12, 20)):
        inputs = self.build_inputs(image, question).to(self.device)
        ids = inputs["input_ids"][0]; L = ids.numel()
        vis_pos = (ids == self.image_token_id).nonzero().flatten()
        q_start = int((ids == self.vision_end_id).nonzero().flatten()[-1]) + 1
        suf = self.proc.tokenizer(PROMPT_SUFFIX.strip(), add_special_tokens=False)["input_ids"]; seq = ids.tolist()
        q_end = next((i for i in range(q_start, L - len(suf) + 1) if seq[i:i + len(suf)] == suf), L)   # suffix start
        dec = self.proc.tokenizer.decode(ids[q_start:q_end]).strip()
        if dec != question.strip():          # fallback: every token after the image
            q_end = L
        q_only = torch.arange(q_start, q_end, device=ids.device); q_all = torch.arange(q_start, L, device=ids.device)
        torch.mps.synchronize(); t0 = time.perf_counter()
        out = self.model(**inputs, output_attentions=True, use_cache=False)
        torch.mps.synchronize(); t_full = time.perf_counter() - t0
        n_vis = int(vis_pos.numel())
        def dist(layer, qpos):
            a = out.attentions[layer][0][:, qpos][:, :, vis_pos].float().mean(dim=(0, 1)); return (a / a.sum()).cpu().numpy()
        def ent(p): return float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(n_vis)) if n_vis > 1 else 1.0
        p = dist(self.k, q_only)                                   # ranking + paper's H: layer k, question tokens only
        p_var = {f"L{ly}_q": dist(ly, q_only) for ly in layers}
        H_var = {}
        for ly in layers:
            H_var[f"L{ly}_q"] = ent(dist(ly, q_only)); H_var[f"L{ly}_all"] = ent(dist(ly, q_all))
        top10 = float(np.sort(p)[::-1][: max(1, n_vis // 10)].sum())    # mass on the top-10% tokens (peakedness)
        t_score = None
        if time_it:
            def stop(*_): raise _Stop()
            h = self._layers[self.k].register_forward_hook(stop)
            try:
                torch.mps.synchronize(); t0 = time.perf_counter(); self.model(**inputs, use_cache=False)
            except _Stop: pass
            finally:
                h.remove(); torch.mps.synchronize(); t_score = time.perf_counter() - t0
        n_txt = int(len(self.proc.tokenizer(question)["input_ids"]))
        return dict(p=p, p_var=p_var, H=ent(p), H_var=H_var, top10=top10, n_vis=n_vis, n_txt=n_txt, q_tokens=int(q_end - q_start),
                    q_exact=(dec == question.strip()), grid_thw=inputs["image_grid_thw"][0].tolist(), t_full=t_full, t_score=t_score, n_seq=int(L))

    # analytic FLOPs ratio r = (ViT + K+1 LLM layers) / (ViT + all LLM layers), per visual token dominant terms
    def flops_ratio(self):
        cfg = self.model.config; v = cfg.vision_config; t = cfg.text_config
        n_layers = t.num_hidden_layers
        # ViT processes 4 patches per visual token (2x2 merge); ~2*params_per_layer*depth per patch
        vit_per_tok = 4 * 2 * (12 * v.hidden_size ** 2) * v.depth
        llm_per_layer = 2 * (4 * t.hidden_size ** 2 + 3 * t.hidden_size * t.intermediate_size)  # attn (approx) + SwiGLU MLP
        full = vit_per_tok + llm_per_layer * n_layers; part = vit_per_tok + llm_per_layer * (self.k + 1)
        return part / full

if __name__ == "__main__":
    import sys, json
    s = Scorer()
    rec = json.loads(open("data/questions.jsonl").readline())
    r = s.score(Image.open(rec["image_path"]).convert("RGB"), rec["question"])
    top = np.argsort(-r["p"])[:5]
    print({k: (v if k != "p" else f"top5 idx {top.tolist()} mass {r['p'][top].sum():.2f}") for k, v in r.items()})
    print("analytic FLOPs ratio r =", round(s.flops_ratio(), 3))
