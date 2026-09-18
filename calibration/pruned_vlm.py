"""Satellite-side VLM (Qwen3-VL-32B, MLX 4-bit) with importance-ranked visual-token retention.

Pruning = gather: the vision tower encodes the full image; before the LLM we keep the
top-ceil(eta*n_vis) visual tokens by an externally supplied importance vector and drop the
rest from inputs_embeds / input_ids / mRoPE position ids / DeepStack rows, preserving the
original positions of the kept tokens (FastV-style). eta=1 must reproduce the unpruned answer.
"""
import math, numpy as np, mlx.core as mx
from PIL import Image
from imgprep import canonical
from mlx_vlm import load
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.base import InputEmbeddingsFeatures

MODEL_ID = "mlx-community/Qwen3-VL-32B-Instruct-4bit"
MIN_PIX, MAX_PIX = 256 * 32 * 32, 1024 * 32 * 32
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

class PrunedVLM:
    def __init__(self, model_id=MODEL_ID):
        self.model, self.processor = load(model_id, processor_kwargs={"min_pixels": MIN_PIX, "max_pixels": MAX_PIX})
        self.lm = self.model.language_model
        self.image_token_id = self.model.config.image_token_index
        tok = self.processor.tokenizer
        self.eos_ids = {tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())

    def build_inputs(self, image: Image.Image, question: str):
        image = canonical(image)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + PROMPT_SUFFIX}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[text], images=[image], return_tensors="np")
        return {k: mx.array(v) for k, v in enc.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    def _features(self, inp, keep_vis=None):
        """Full features from the library, then gather the kept sequence positions."""
        feats = self.model.get_input_embeddings(inp["input_ids"], inp["pixel_values"],
                                                image_grid_thw=inp["image_grid_thw"], mask=inp["attention_mask"])
        ids = np.array(inp["input_ids"][0])
        vis_seq = np.where(ids == self.image_token_id)[0]                # sequence positions of visual tokens
        n_vis, L = len(vis_seq), len(ids)
        if keep_vis is None: return feats, inp["input_ids"], n_vis
        keep_vis = np.sort(np.asarray(keep_vis, dtype=np.int64)); assert keep_vis.max() < n_vis
        drop_seq = set(vis_seq[np.setdiff1d(np.arange(n_vis), keep_vis)].tolist())
        keep_seq = np.array([i for i in range(L) if i not in drop_seq]); ks = mx.array(keep_seq); kv = mx.array(keep_vis)
        pos = feats.position_ids
        pos = pos[:, :, ks] if pos.ndim == 3 else pos[:, ks]
        ds = feats.deepstack_visual_embeds
        if ds is not None:                                                # per-layer (n_vis, D) arrays -> kept rows, in order
            ds = [layer[kv, :] for layer in ds] if isinstance(ds, (list, tuple)) else ds[:, kv, :]
        new = InputEmbeddingsFeatures(
            inputs_embeds=feats.inputs_embeds[:, ks, :],
            visual_pos_masks=feats.visual_pos_masks[:, ks] if feats.visual_pos_masks is not None else None,
            deepstack_visual_embeds=ds, position_ids=pos,
            rope_deltas=feats.rope_deltas + (L - len(keep_seq)) if feats.rope_deltas is not None else None)
        return new, mx.array(ids[keep_seq])[None, :], n_vis

    def answer(self, image: Image.Image, question: str, importance=None, eta=1.0, max_new_tokens=10, keep=None):
        inp = self.build_inputs(image, question)
        ids = np.array(inp["input_ids"][0]); n_vis = int((ids == self.image_token_id).sum())
        if keep is not None:
            keep = np.asarray(keep); assert keep.max() < n_vis and len(keep) == len(set(keep.tolist()))
        elif eta < 1.0:
            assert importance is not None and len(importance) == n_vis, f"importance len {None if importance is None else len(importance)} != n_vis {n_vis}"
            k = max(1, math.ceil(eta * n_vis)); keep = np.argsort(-np.asarray(importance))[:k]
        self.lm._rope_deltas = None; self.lm._position_ids = None
        feats, pruned_ids, _ = self._features(inp, keep)
        kv = vlm_cache.make_prompt_cache(self.lm)
        kw = dict(position_ids=feats.position_ids, rope_deltas=feats.rope_deltas,
                  visual_pos_masks=feats.visual_pos_masks, deepstack_visual_embeds=feats.deepstack_visual_embeds)
        out = self.lm(pruned_ids, inputs_embeds=feats.inputs_embeds, cache=kv, **kw)
        toks = []
        y = mx.argmax(out.logits[:, -1, :], axis=-1)
        for _ in range(max_new_tokens):
            t = int(y.item())
            if t in self.eos_ids: break
            toks.append(t)
            out = self.lm(y[None], cache=kv, rope_deltas=feats.rope_deltas)   # decode: positions from cache offset + rope_deltas
            y = mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.clear_cache()
        return self.processor.tokenizer.decode(toks).strip(), dict(n_vis=n_vis, n_kept=(len(keep) if keep is not None else n_vis),
                                                                     n_seq=int(pruned_ids.shape[1]), grid_thw=np.array(inp["image_grid_thw"][0]).tolist())

    def answer_timed(self, image, question, keep=None, eta=1.0, max_new_tokens=20):
        """Same as answer() but returns (t_prefill, [decode step times]) measured with mx.eval barriers."""
        import time
        inp = self.build_inputs(image, question); ids = np.array(inp["input_ids"][0]); n_vis = int((ids == self.image_token_id).sum())
        if keep is None and eta < 1.0: keep = np.arange(n_vis)[: max(1, math.ceil(eta * n_vis))]
        self.lm._rope_deltas = None; self.lm._position_ids = None
        t0 = time.perf_counter(); feats, pruned_ids, _ = self._features(inp, keep); kv = vlm_cache.make_prompt_cache(self.lm)
        kw = dict(position_ids=feats.position_ids, rope_deltas=feats.rope_deltas, visual_pos_masks=feats.visual_pos_masks, deepstack_visual_embeds=feats.deepstack_visual_embeds)
        out = self.lm(pruned_ids, inputs_embeds=feats.inputs_embeds, cache=kv, **kw); y = mx.argmax(out.logits[:, -1, :], axis=-1); mx.eval(y)
        t_prefill = time.perf_counter() - t0; steps = []
        for _ in range(max_new_tokens):
            t1 = time.perf_counter(); out = self.lm(y[None], cache=kv, rope_deltas=feats.rope_deltas); y = mx.argmax(out.logits[:, -1, :], axis=-1); mx.eval(y); steps.append(time.perf_counter() - t1)
        mx.clear_cache(); return t_prefill, steps, int(pruned_ids.shape[1]), n_vis

if __name__ == "__main__":
    import json, time
    from mlx_vlm import generate as lib_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    v = PrunedVLM(); rec = json.loads(open("data/questions.jsonl").readline()); img = Image.open(rec["image_path"]).convert("RGB")
    print("Q:", rec["question"], "| GT:", rec["answers"][:3])
    # reference: library generate (unpruned)
    prompt = apply_chat_template(v.processor, v.model.config, rec["question"] + PROMPT_SUFFIX, num_images=1)
    t0 = time.time(); ref = lib_generate(v.model, v.processor, prompt, image=[img], max_tokens=10, temperature=0.0, verbose=False)
    print(f"library  : {getattr(ref,'text',ref)!r}  ({time.time()-t0:.1f}s)")
    t0 = time.time(); a1, m = v.answer(img, rec["question"], eta=1.0); print(f"ours eta=1: {a1!r}  {m}  ({time.time()-t0:.1f}s)")
    rng = np.random.default_rng(0); imp = rng.random(m["n_vis"])
    for eta in (0.5, 0.2):
        t0 = time.time(); a, m2 = v.answer(img, rec["question"], importance=imp, eta=eta); print(f"ours eta={eta}: {a!r}  kept {m2['n_kept']}/{m2['n_vis']} seq {m2['n_seq']}  ({time.time()-t0:.1f}s)")
