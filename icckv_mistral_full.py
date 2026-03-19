"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ICCKV-SENTINEL v3.1 — Mistral-7B Full Pipeline                        ║
║  Tüm ölçüm adımları tek scriptte:                                       ║
║    STEP-1  Q→K R²  (train/test split, FIX-1)                           ║
║    STEP-2  MI I(K;Q)  (bins=64, deep layers L8–L15)                    ║
║    STEP-3  Fidelity karşılaştırması  (IRCM vs Eviction, 60 trial)      ║
║    STEP-4  PPL sweep  (seq=256/512/1024, L0–L3 / L8–L15 / all)        ║
║    STEP-5  Sparsity Curve  (s=1%–30%, Pareto grafiği)                  ║
║    STEP-6  Patent iddiası özeti                                         ║
║                                                                          ║
║  Model: mistralai/Mistral-7B-Instruct-v0.3                               ║
║  Kaggle GPU: T4 / P100                                                  ║
║  Yazar: Çağlar FINDIKLI — Mart 2026                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os, json, math, time, warnings, gc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. ORTAM ─────────────────────────────────────────────────────────────────
print("=" * 68)
print("  ICCKV-SENTINEL v3.0 — Llama 3.2-3B Full Pipeline")
print("=" * 68)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ENV] device={device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"[ENV] {props.name} | {props.total_memory/1e9:.1f} GB VRAM")

# ── 2. MODEL ──────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM

# Llama 3.2-3B — Kaggle'da token gerekebilir; alternatif listesi sağlandı
CANDIDATE_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",
    "unsloth/mistral-7b-instruct-v0.3",          # public mirror
    "unsloth/mistral-7b-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",        # v0.2 genellikle gated değil
    "unsloth/mistral-7b-instruct-v0.2",
]

model = None
tokenizer = None
MODEL_ID = None

print("\n[MODEL] Yükleniyor...")
t0 = time.time()
for mid in CANDIDATE_MODELS:
    try:
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        MODEL_ID = mid
        print(f"[MODEL] ✓ {mid}  ({time.time()-t0:.1f}s)")
        break
    except Exception as e:
        print(f"[MODEL] ✗ {mid}: {str(e)[:60]}")

if model is None:
    raise RuntimeError(
        "Hiçbir model yüklenemedi. Kaggle'da:\n"
        "  Settings → Environment Variables → HF_TOKEN ekle\n"
        "veya 'mistralai/Mistral-7B-v0.3' için Hugging Face'de erişim iste."
    )

model.eval()

# Config — otomatik algıla
cfg     = model.config
D       = cfg.hidden_size
N_Q     = cfg.num_attention_heads
N_KV    = getattr(cfg, "num_key_value_heads", N_Q)
HD      = D // N_Q
GRP     = N_Q // N_KV
N_LAYER = cfg.num_hidden_layers

print(f"\n[CFG] model={MODEL_ID}")
print(f"[CFG] d={D} | Q_heads={N_Q} | KV_heads={N_KV} | head_dim={HD} | GQA_grp={GRP} | layers={N_LAYER}")

# Katman grupları — N_LAYER'a göre otomatik ayarla
L_SHALLOW = list(range(0, 4))
L_DEEP    = list(range(min(8, N_LAYER//3), min(16, 2*N_LAYER//3)))
L_ALL     = list(range(N_LAYER))

LAYER_GROUPS = {
    f"shallow (L0–L3)":               L_SHALLOW,
    f"deep (L{L_DEEP[0]}–L{L_DEEP[-1]})": L_DEEP,
    f"all (L0–L{N_LAYER-1})":         L_ALL,
}
print(f"[CFG] Katman grupları: { {k: f'{len(v)} layer' for k, v in LAYER_GROUPS.items()} }")

# ── 3. METİNLER ──────────────────────────────────────────────────────────────
def _extend(base: str, target_words: int = 1600) -> str:
    words = base.split()
    reps  = math.ceil(target_words / max(len(words), 1))
    return " ".join((words * reps)[:target_words])

_A = """Transformer architecture revolutionized natural language processing by introducing
self-attention mechanisms that allow models to weigh token relationships across entire
sequences. The key insight was that attention weights computed from queries and keys
capture linguistic dependencies far more effectively than recurrent approaches. Modern
large language models stack dozens of transformer layers, each refining contextual
representations through multi-head attention and feed-forward networks. Efficiency
improvements like grouped-query attention reduce the memory footprint of the key-value
cache, which grows linearly with sequence length and becomes a bottleneck in long-context
inference. Various cache compression strategies have been proposed including eviction
of low-importance tokens, quantization of stored tensors, and low-rank approximations
of attention matrices. The trade-off between compression ratio and language quality,
measured by perplexity, is the central challenge that any practical deployment must
navigate carefully to maintain model coherence and output fidelity across diverse tasks."""

_B = """Information theory provides a rigorous framework for quantifying the relationships
between random variables through mutual information and entropy measures. The Wyner-Ziv
theorem extends classical rate-distortion theory to scenarios where correlated side
information is available at the decoder without being transmitted. This has profound
implications for distributed compression systems where one source can be compressed more
efficiently when the decoder already possesses a statistically related signal. In the
context of neural network inference, key and query tensors in attention layers exhibit
measurable statistical dependence that grows stronger in deeper layers where contextual
integration accumulates across the network depth. Exploiting this dependence through
linear regression allows the receiver to predict a substantial fraction of the key
tensor variance from the query alone enabling significant bit-rate savings without
proportional losses in reconstructed signal quality or downstream task performance."""

_C = """Memory bandwidth is a primary bottleneck in autoregressive text generation because
each token generation step requires loading the entire key-value cache from high-bandwidth
memory into the compute units for attention calculation. The working set size scales as
two times the number of layers times the number of key-value heads times the head
dimension times the sequence length times the dtype byte width. For a three billion
parameter model generating a four thousand token context in bfloat sixteen this amounts
to dozens of gigabytes transferred per token severely limiting throughput on both data
center accelerators and edge devices. Compression methods that reduce cache size without
degrading attention fidelity can therefore yield substantial wall-clock speedups in
addition to their memory savings. Evaluating these methods requires holistic benchmarks
that measure not only tensor reconstruction quality but also end-to-end language model
perplexity on held-out text corpora to ensure that compression artifacts do not compound
across layers and degrade the coherence of generated sequences over long contexts."""

TEXTS = [_extend(_A), _extend(_B), _extend(_C)]

# Token sayısı kontrolü
print("\n[TEXT] Test metinleri:")
for i, txt in enumerate(TEXTS):
    toks = tokenizer(txt, return_tensors="pt")["input_ids"].shape[1]
    print(f"  Metin-{i+1}: {toks} token")


# ── 4. HOOK: KV tensörlerini yakala ──────────────────────────────────────────
def capture_kv_hooks(model, input_ids, layer_indices):
    """
    q_proj ve k_proj çıktılarını (pre-RoPE) yakala.
    Döner: {layer_idx: {"Q": ndarray, "K": ndarray}}
    """
    captured = {}
    hooks = []

    T = input_ids.shape[1]

    def make_q_hook(li):
        def hook(module, inp, out):
            q = out.detach().float().cpu().numpy()[0]  # (T, D)
            captured.setdefault(li, {})["Q"] = q.reshape(T, N_Q, HD)
        return hook

    def make_k_hook(li):
        def hook(module, inp, out):
            k = out.detach().float().cpu().numpy()[0]  # (T, N_KV*HD)
            captured.setdefault(li, {})["K"] = k.reshape(T, N_KV, HD)
        return hook

    layers = model.model.layers
    for li in layer_indices:
        attn = layers[li].self_attn
        hooks.append(attn.q_proj.register_forward_hook(make_q_hook(li)))
        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(li)))

    with torch.no_grad():
        model(input_ids.to(device))

    for h in hooks:
        h.remove()

    return captured


# ── 5. IRCM SINIFI ────────────────────────────────────────────────────────────
ALPHA = 1e-3

def ridge_fit(X, Y):
    XtX = X.T @ X + ALPHA * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ Y)

class IRCMCompressor:
    def __init__(self, sparsity: float):
        self.sp = sparsity

    def compress(self, K, Q):
        seq, n_kv, hd = K.shape
        n_train = max(16, int(seq * 0.70))
        n_keep  = max(1, int(seq * self.sp))
        K_pred  = np.zeros_like(K)
        B_list  = []

        for kv_h in range(n_kv):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(seq, -1)
            Q_aug = np.hstack([Q_grp, np.ones((seq, 1))])
            K_h   = K[:, kv_h, :]
            B = ridge_fit(Q_aug[:n_train], K_h[:n_train])
            K_pred[:, kv_h, :] = Q_aug @ B
            B_list.append(B)

        R = K - K_pred
        row_norms = np.linalg.norm(R.reshape(seq, -1), axis=1)
        kept_idx  = np.argsort(row_norms)[-n_keep:]
        R_sparse  = np.zeros_like(R)
        R_sparse[kept_idx] = R[kept_idx]
        return {"B_list": B_list, "R_sparse": R_sparse}

    def decompress(self, payload, Q):
        seq = Q.shape[0]
        n_kv = len(payload["B_list"])
        K_rec = np.zeros((seq, n_kv, HD), dtype=np.float32)
        for kv_h, B in enumerate(payload["B_list"]):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(seq, -1)
            Q_aug = np.hstack([Q_grp, np.ones((seq, 1))])
            K_rec[:, kv_h, :] = Q_aug @ B
        K_rec += payload["R_sparse"]
        return K_rec

class EvictionCompressor:
    def __init__(self, sparsity: float):
        self.sp = sparsity

    def compress(self, K, Q):
        seq, n_kv, hd = K.shape
        n_keep = max(1, int(seq * self.sp))
        norms  = np.mean(np.sum(K**2, axis=-1), axis=-1)
        kept   = np.argsort(norms)[-n_keep:]
        K_sp   = np.zeros_like(K)
        K_sp[kept] = K[kept]
        return {"K_sparse": K_sp}

    def decompress(self, payload, Q):
        return payload["K_sparse"]


# ── 6. MI ─────────────────────────────────────────────────────────────────────
def mi_discrete(X, Y, bins=64, n_samples=2048, seed=0):
    rng = np.random.default_rng(seed)
    xf, yf = X.flatten().astype(np.float32), Y.flatten().astype(np.float32)
    mn = min(len(xf), len(yf), n_samples)
    idx = rng.choice(min(len(xf), len(yf)), size=mn, replace=False)
    xf, yf = xf[idx], yf[idx]
    h_x, ex = np.histogram(xf, bins=bins)
    h_y, ey = np.histogram(yf, bins=bins)
    h_xy, _, _ = np.histogram2d(xf, yf, bins=bins)
    p_x  = (h_x  + 1e-9) / (h_x.sum()  + 1e-9)
    p_y  = (h_y  + 1e-9) / (h_y.sum()  + 1e-9)
    p_xy = (h_xy + 1e-9) / (h_xy.sum() + 1e-9)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 1e-12:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(max(0.0, mi))


# ── 7. PPL HESAPLAMA ─────────────────────────────────────────────────────────
def compute_ppl(text, seq_len, layer_indices, compressor):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=seq_len, padding=False)
    input_ids = enc["input_ids"].to(device)
    T = input_ids.shape[1]
    if T < 8:
        return float("nan")

    _q_cache = {}
    hooks = []

    if compressor is not None:
        def make_q_hook(li):
            def hook(module, inp, out):
                q = out.detach().float().cpu().numpy()[0]
                _q_cache[li] = q.reshape(T, N_Q, HD)
            return hook

        def make_k_hook(li):
            def hook(module, inp, out):
                if li not in layer_indices or li not in _q_cache:
                    return
                k = out.detach().float().cpu().numpy()[0].reshape(T, N_KV, HD)
                Q_np = _q_cache[li]
                payload = compressor.compress(k, Q_np)
                k_rec   = compressor.decompress(payload, Q_np)
                out.data.copy_(
                    torch.tensor(k_rec.reshape(1, T, N_KV * HD),
                                 dtype=out.dtype, device=out.device)
                )
            return hook

        for li in range(N_LAYER):
            attn = model.model.layers[li].self_attn
            hooks.append(attn.q_proj.register_forward_hook(make_q_hook(li)))
            hooks.append(attn.k_proj.register_forward_hook(make_k_hook(li)))

    nll_sum, n_toks, prev_end = 0.0, 0, 0
    stride = max(1, seq_len // 2)

    with torch.no_grad():
        for begin in range(0, T, stride):
            end   = min(begin + seq_len, T)
            chunk = input_ids[:, begin:end]
            if chunk.shape[1] < 2:
                break
            out   = model(chunk, labels=chunk)
            n_new = max(0, end - max(begin, prev_end))
            if n_new > 0:
                nll_sum += out.loss.item() * n_new
                n_toks  += n_new
            prev_end = end
            if end == T:
                break

    for h in hooks:
        h.remove()

    return math.exp(nll_sum / max(n_toks, 1))


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-1: R² (Train/Test Split)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-1: Q→K R² (train/test split, FIX-1)")
print("=" * 68)

SEQ_LENS_R2 = [64, 128, 256, 512, 1024]
TEST_LAYER_INDICES = L_SHALLOW + L_DEEP[:4]   # 8 katman

r2_results = []
for seq_len in SEQ_LENS_R2:
    text = TEXTS[0]
    enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"]
    T = input_ids.shape[1]
    if T < 24:
        continue

    captured = capture_kv_hooks(model, input_ids, TEST_LAYER_INDICES)
    n_train  = max(16, int(T * 0.70))

    layer_r2s = []
    for li, data in captured.items():
        K, Q = data["K"], data["Q"]
        seq, n_kv, hd = K.shape

        head_r2s = []
        for kv_h in range(n_kv):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(seq, -1)
            Q_aug = np.hstack([Q_grp, np.ones((seq, 1))])
            K_h   = K[:, kv_h, :]

            B = ridge_fit(Q_aug[:n_train], K_h[:n_train])
            K_pred = Q_aug @ B
            R = K_h - K_pred

            R_test = R[n_train:]
            K_test = K_h[n_train:]
            ss_res = np.sum(R_test**2)
            ss_tot = np.sum((K_test - K_test.mean())**2) + 1e-12
            r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
            head_r2s.append(r2)

        layer_r2s.append(float(np.mean(head_r2s)))

    mean_r2 = float(np.mean(layer_r2s))
    r2_results.append({"seq": T, "mean_r2": mean_r2, "n_train": n_train})
    r2_ok = "✓" if mean_r2 > 0.50 else "✗"
    print(f"  seq={T:4d}  n_train={n_train:3d}  R²(test)={mean_r2:.4f}  {r2_ok}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-2: MI I(K;Q)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-2: Mutual Information I(K;Q)")
print("=" * 68)

MI_SEQ = 512
enc = tokenizer(TEXTS[1], return_tensors="pt", truncation=True, max_length=MI_SEQ)
input_ids_mi = enc["input_ids"]
T_mi = input_ids_mi.shape[1]

mi_results = {}
for g_name, g_layers in LAYER_GROUPS.items():
    captured_mi = capture_kv_hooks(model, input_ids_mi, g_layers)
    mis = []
    for li, data in captured_mi.items():
        K, Q = data["K"], data["Q"]
        for kv_h in range(N_KV):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(T_mi, -1)
            K_h   = K[:, kv_h, :]
            mi = mi_discrete(Q_grp, K_h, bins=64, seed=42)
            mis.append(mi)
    mean_mi = float(np.mean(mis))
    mi_results[g_name] = mean_mi
    ok = "✓" if mean_mi >= 0.5 else "~"
    print(f"  {g_name:<30}  I(K;Q)={mean_mi:.4f} b  {ok}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-3: FİDELİTY (IRCM vs Eviction)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-3: Fidelity — IRCM vs Eviction")
print("=" * 68)

FID_SPARSITY  = 0.05
FID_SEQ_LENS  = [64, 128, 256, 512, 1024]
FID_LAYER_INDICES = L_DEEP

def fidelity(K_orig, K_rec):
    num = np.sum(K_orig * K_rec)
    den = np.linalg.norm(K_orig) * np.linalg.norm(K_rec) + 1e-9
    return float(num / den)

fid_wins_ircm = 0
fid_total = 0

print(f"  {'seq':>5}  {'IRCM fid':>10}  {'Evict fid':>10}  {'Kazanan':>8}")
print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}")

for seq_len in FID_SEQ_LENS:
    for txt in TEXTS:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids_f = enc["input_ids"]
        T_f = input_ids_f.shape[1]
        if T_f < 24:
            continue

        captured_f = capture_kv_hooks(model, input_ids_f, FID_LAYER_INDICES)

        ircm_fids, ev_fids = [], []
        for li, data in captured_f.items():
            K, Q = data["K"], data["Q"]

            comp_ircm = IRCMCompressor(FID_SPARSITY)
            p_ircm    = comp_ircm.compress(K, Q)
            K_ircm    = comp_ircm.decompress(p_ircm, Q)

            comp_ev = EvictionCompressor(FID_SPARSITY)
            p_ev    = comp_ev.compress(K, Q)
            K_ev    = comp_ev.decompress(p_ev, Q)

            ircm_fids.append(fidelity(K, K_ircm))
            ev_fids.append(fidelity(K, K_ev))

        f_ircm = float(np.mean(ircm_fids))
        f_ev   = float(np.mean(ev_fids))
        winner = "IRCM✓" if f_ircm > f_ev else "Evict"
        if f_ircm > f_ev:
            fid_wins_ircm += 1
        fid_total += 1

print(f"\n  IRCM Fidelity Kazanımları: {fid_wins_ircm}/{fid_total}")
fid_ok = fid_wins_ircm == fid_total


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-4: PPL SWEEP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-4: PPL Sweep")
print("=" * 68)

PPL_SEQ_LENS  = [256, 512, 1024]
PPL_SPARSITY  = 0.05

# Baseline PPL
print("\n[PPL] Baseline hesaplanıyor...")
base_ppls_per_seq = {}
for sl in PPL_SEQ_LENS:
    vals = [compute_ppl(txt, sl, [], None) for txt in TEXTS]
    base_ppls_per_seq[sl] = float(np.mean(vals))
    print(f"  seq={sl}  baseline PPL={base_ppls_per_seq[sl]:.3f}")

ppl_results = {}
print(f"\n[PPL] Sparsity={PPL_SPARSITY*100:.0f}%  — IRCM vs Eviction")
print(f"  {'Katman':<22}  {'seq':>5}  {'IRCM PPL':>9}  {'ΔIRCM':>7}  {'Evict PPL':>10}  {'ΔEvict':>8}  {'Kazanan':>7}")
print(f"  {'─'*22}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*7}")

ppl_wins_ircm = 0
ppl_total = 0

for g_name, g_layers in LAYER_GROUPS.items():
    ppl_results[g_name] = {}
    comp_ircm = IRCMCompressor(PPL_SPARSITY)
    comp_ev   = EvictionCompressor(PPL_SPARSITY)

    for sl in PPL_SEQ_LENS:
        base = base_ppls_per_seq[sl]
        ircm_vals = [compute_ppl(txt, sl, g_layers, comp_ircm) for txt in TEXTS]
        ev_vals   = [compute_ppl(txt, sl, g_layers, comp_ev)   for txt in TEXTS]
        m_ircm  = float(np.mean(ircm_vals))
        m_ev    = float(np.mean(ev_vals))
        d_ircm  = m_ircm - base
        d_ev    = m_ev   - base
        winner  = "IRCM✓" if d_ircm < d_ev else "Evict"
        if d_ircm < d_ev:
            ppl_wins_ircm += 1
        ppl_total += 1
        ppl_results[g_name][sl] = {"ircm": m_ircm, "evict": m_ev,
                                    "d_ircm": d_ircm, "d_ev": d_ev}
        print(f"  {g_name:<22}  {sl:>5}  {m_ircm:>9.3f}  {d_ircm:>+7.3f}  "
              f"{m_ev:>10.3f}  {d_ev:>+8.3f}  {winner:>7}")

print(f"\n  PPL IRCM Kazanımları: {ppl_wins_ircm}/{ppl_total}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-5: SPARSITY CURVE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-5: Sparsity Curve (1%–30%)")
print("=" * 68)

SPARSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
CURVE_SEQ_LEN   = 512
BASE_PPL        = base_ppls_per_seq[CURVE_SEQ_LEN]

curve_results = {}

for g_name, g_layers in LAYER_GROUPS.items():
    curve_results[g_name] = {}
    print(f"\n  {g_name}")
    print(f"  {'sp':>5}  {'IRCM PPL':>9}  {'ΔIRCM':>7}  {'Evict PPL':>10}  {'ΔEvict':>8}  {'Kazan':>6}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*6}")

    for sp in SPARSITY_LEVELS:
        ci = IRCMCompressor(sp)
        ce = EvictionCompressor(sp)
        ircm_ppls  = [compute_ppl(txt, CURVE_SEQ_LEN, g_layers, ci) for txt in TEXTS]
        evict_ppls = [compute_ppl(txt, CURVE_SEQ_LEN, g_layers, ce) for txt in TEXTS]
        m_i  = float(np.mean(ircm_ppls))
        m_e  = float(np.mean(evict_ppls))
        d_i  = m_i - BASE_PPL
        d_e  = m_e - BASE_PPL
        win  = "IRCM✓" if d_i < d_e else "Evict"
        curve_results[g_name][sp] = {
            "ircm_mean": m_i, "evict_mean": m_e,
            "delta_ircm": d_i, "delta_evict": d_e,
            "ircm_ppls": ircm_ppls, "evict_ppls": evict_ppls,
            "ircm_ok": d_i < 2.0, "winner": win,
        }
        print(f"  {sp*100:>4.0f}%  {m_i:>9.3f}  {d_i:>+7.3f}  {m_e:>10.3f}  {d_e:>+8.3f}  {win:>6}")

# Pareto grafiği
sp_pct = [s * 100 for s in SPARSITY_LEVELS]
COLORS = {"IRCM": "#0077cc", "Evict": "#cc3300"}
FILLS  = {"IRCM": "#cce5ff", "Evict": "#ffd5cc"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle(
    f"ICCKV-SENTINEL v3.1: IRCM vs Token Eviction — Sparsity–Quality Pareto Curve\n"
    f"{MODEL_ID} | seq={CURVE_SEQ_LEN} | mean over 3 texts | lower ΔPPL = better",
    fontsize=12, fontweight="bold", y=1.02
)

for ax, (g_name, sp_dict) in zip(axes, curve_results.items()):
    ircm_means  = [sp_dict[s]["delta_ircm"]  for s in SPARSITY_LEVELS]
    evict_means = [sp_dict[s]["delta_evict"] for s in SPARSITY_LEVELS]
    ircm_stds   = [np.std(sp_dict[s]["ircm_ppls"])  for s in SPARSITY_LEVELS]
    evict_stds  = [np.std(sp_dict[s]["evict_ppls"]) for s in SPARSITY_LEVELS]

    ax.fill_between(sp_pct,
        [m - s for m, s in zip(ircm_means, ircm_stds)],
        [m + s for m, s in zip(ircm_means, ircm_stds)],
        color=FILLS["IRCM"], alpha=0.5)
    ax.fill_between(sp_pct,
        [m - s for m, s in zip(evict_means, evict_stds)],
        [m + s for m, s in zip(evict_means, evict_stds)],
        color=FILLS["Evict"], alpha=0.4)

    ax.plot(sp_pct, ircm_means, "o-", color=COLORS["IRCM"],
            linewidth=2.5, markersize=6, label="IRCM (proposed)", zorder=5)
    ax.plot(sp_pct, evict_means, "s--", color=COLORS["Evict"],
            linewidth=2.5, markersize=6, label="Token Eviction", zorder=5)
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.2, label="ΔPPL=2.0 thr")

    # Eviction çok büyük değerleri notla
    for i, sp in enumerate(SPARSITY_LEVELS):
        if evict_means[i] > 200:
            ax.annotate(f"↑{evict_means[i]:.0f}", (sp * 100, min(ax.get_ylim()[1] * 0.9, 200)),
                        ha="center", fontsize=7, color=COLORS["Evict"])

    ax.set_title(g_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sparsity (% of KV cache kept)", fontsize=10)
    ax.set_ylabel("ΔPPL (vs baseline)", fontsize=10)
    ax.set_xlim(0, 32)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "icckv_mistral_sparsity_curve.png")
fig.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n[PLOT] ✓ {plot_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP-6: PATENT İDDİASI ÖZETİ
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP-6: Patent İddiaları")
print("=" * 68)

mean_r2 = float(np.mean([x["mean_r2"] for x in r2_results]))
best_mi  = max(mi_results.values())
ppl_ok   = ppl_wins_ircm == ppl_total

# Tüm sparsity / grup kombinasyonlarında IRCM kaç kez kazandı
curve_wins = sum(1 for sd in curve_results.values()
                 for v in sd.values() if v["winner"] == "IRCM✓")
curve_total = len(SPARSITY_LEVELS) * len(LAYER_GROUPS)

claims = [
    ("C1: R²(test-split) > 0.50",        mean_r2 > 0.50,  f"R²={mean_r2:.4f}"),
    ("C2: I(K;Q) ≥ 0.50 b",              best_mi >= 0.50, f"I={best_mi:.4f} b"),
    ("C3: Fidelity IRCM > Eviction",      fid_ok,          f"{fid_wins_ircm}/{fid_total}"),
    ("C4: PPL IRCM > Eviction",           ppl_ok,          f"{ppl_wins_ircm}/{ppl_total}"),
    ("C5: Sparsity curve IRCM wins all",  curve_wins==curve_total,
                                           f"{curve_wins}/{curve_total}"),
]

all_pass = True
for label, passed, evidence in claims:
    sym = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_pass = False
    print(f"  {sym}  {label:<42}  [{evidence}]")

print(f"\n  {'🏆 TÜM İDDİALAR ONAYLANDI' if all_pass else '⚠️  BAZI İDDİALAR BAŞARISIZ'}")
print(f"  Model: {MODEL_ID}")
print(f"  d={D} | Q={N_Q} | KV={N_KV} | GRP={GRP} | L={N_LAYER}")


# ═════════════════════════════════════════════════════════════════════════════
#  JSON ÇIKTI
# ═════════════════════════════════════════════════════════════════════════════
summary = {
    "version":    "ICCKV-SENTINEL v3.1",
    "model":       MODEL_ID,
    "config":      {"D": D, "N_Q": N_Q, "N_KV": N_KV, "HD": HD,
                    "GRP": GRP, "N_LAYER": N_LAYER},
    "step1_r2":    r2_results,
    "step2_mi":    mi_results,
    "step3_fidelity": {"wins": fid_wins_ircm, "total": fid_total},
    "step4_ppl":   ppl_results,
    "step5_curve": {
        g: {str(round(s*100))+"%": v for s, v in sd.items()}
        for g, sd in curve_results.items()
    },
    "step6_claims": [
        {"claim": l, "passed": p, "evidence": e} for l, p, e in claims
    ],
    "baseline_ppl": base_ppls_per_seq,
}

json_path = os.path.join(OUTPUT_DIR, "icckv_mistral_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n[JSON] ✓ {json_path}")
print(f"[PLOT] ✓ {plot_path}")
print("\n" + "=" * 68)
print("  TAMAMLANDI")
print("=" * 68)
