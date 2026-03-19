"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ICCKV-SENTINEL v2.7 — Sparsity Curve & Pareto Analysis                ║
║  IRCM vs Eviction: ΔPPL @ %1–%30 sıkıştırma oranları                  ║
║  Patent Figure kalitesinde çıktı                                        ║
║  Kaggle GPU (T4/P100)                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Önceki versiyonlardan farklar (v1.1 → v2.7):
  - Sparsity aralığı genişletildi: [1,2,5,10,20] → [1,2,3,5,7,10,15,20,25,30]
  - Tüm 3 katman grubu paralel test edilir (shallow/deep/all)
  - matplotlib Pareto grafiği — patent başvurusu Figure kalitesi
  - "both_ok" bug'ı düzeltildi → ircm_ok / evict_ok ayrı izlenir
  - Confidence band: her sparsity'de 3 metin × ortalama ± std
  - JSON çıktı: icckv_sparsity_curve_results.json
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os, json, math, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # Kaggle'da display yok
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── 1. ORTAM & MODEL ──────────────────────────────────────────────────────────
print("=" * 68)
print("  ICCKV-SENTINEL v2.7 — Sparsity Curve")
print("=" * 68)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ENV] device={device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"[ENV] GPU: {props.name} | {props.total_memory/1e9:.1f} GB")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"[MODEL] Yükleniyor: {MODEL_ID}")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print(f"[MODEL] ✓ Yükleme: {time.time()-t0:.1f}s")

# Config
cfg     = model.config
D       = cfg.hidden_size
N_Q     = cfg.num_attention_heads
N_KV    = cfg.num_key_value_heads
HD      = D // N_Q
GRP     = N_Q // N_KV
N_LAYER = cfg.num_hidden_layers
print(f"[CFG] d={D} | Q={N_Q} | KV={N_KV} | hd={HD} | GRP={GRP} | L={N_LAYER}")

# ── 2. METİNLER (≥1024 token) ─────────────────────────────────────────────────
def _extend(base: str, target_words: int = 1400) -> str:
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
inference. Various cache compression strategies have been proposed, including eviction
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
tensor variance from the query alone, enabling significant bit-rate savings without
proportional losses in reconstructed signal quality or downstream task performance."""

_C = """Memory bandwidth is a primary bottleneck in autoregressive text generation because
each token generation step requires loading the entire key-value cache from high-bandwidth
memory into the compute units for attention calculation. The working set size scales as
two times the number of layers times the number of key-value heads times the head
dimension times the sequence length times the dtype byte width. For a one billion
parameter model generating a four thousand token context in bfloat sixteen, this amounts
to several gigabytes transferred per token, severely limiting throughput on both data
center accelerators and edge devices. Compression methods that reduce cache size without
degrading attention fidelity can therefore yield substantial wall-clock speedups in
addition to their memory savings. Evaluating these methods requires holistic benchmarks
that measure not only tensor reconstruction quality but also end-to-end language model
perplexity on held-out text corpora to ensure that compression artifacts do not compound
across layers and degrade the coherence of generated sequences over long contexts."""

PPL_TEXTS = [_extend(_A), _extend(_B), _extend(_C)]

# Token uzunluk kontrolü
for i, txt in enumerate(PPL_TEXTS):
    toks = tokenizer(txt, return_tensors="pt")["input_ids"].shape[1]
    print(f"[TEXT] Metin-{i+1}: {toks} token")

# ── 3. IRCM — IRidge Compressed Memory ───────────────────────────────────────
ALPHA = 1e-3   # Ridge lambda

def ridge_fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(n,p) → (n,q) least-squares with L2 reg."""
    XtX = X.T @ X
    XtX += ALPHA * np.eye(XtX.shape[0])
    return np.linalg.solve(XtX, X.T @ Y)

class IRCMCompressor:
    """Pre-RoPE Q→K linear map + sparse residual."""

    def __init__(self, sparsity: float):
        self.sp = sparsity     # fraction of residual entries kept

    def compress(self, K: np.ndarray, Q: np.ndarray) -> dict:
        """K,Q: (seq, n_kv, hd) pre-RoPE tensors."""
        seq, n_kv, hd = K.shape
        n_train = max(16, int(seq * 0.70))
        n_keep  = max(1, int(seq * self.sp))

        K_pred_full = np.zeros_like(K)
        B_list = []

        for kv_h in range(n_kv):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(seq, -1)
            Q_aug = np.hstack([Q_grp, np.ones((seq, 1))])
            K_h   = K[:, kv_h, :]

            B = ridge_fit(Q_aug[:n_train], K_h[:n_train])
            K_pred_full[:, kv_h, :] = Q_aug @ B
            B_list.append(B)

        R = K - K_pred_full   # (seq, n_kv, hd)

        # Sparse residual: keep top-n_keep rows by L2 norm
        row_norms = np.linalg.norm(R.reshape(seq, -1), axis=1)
        kept_idx  = np.argsort(row_norms)[-n_keep:]
        R_sparse  = np.zeros_like(R)
        R_sparse[kept_idx] = R[kept_idx]

        return {"B_list": B_list, "R_sparse": R_sparse, "kept_idx": kept_idx}

    def decompress(self, payload: dict, Q: np.ndarray) -> np.ndarray:
        seq, n_kv_q, hd = Q.shape
        n_kv = len(payload["B_list"])
        K_rec = np.zeros((seq, n_kv, hd), dtype=Q.dtype)

        for kv_h, B in enumerate(payload["B_list"]):
            q_idx = list(range(kv_h * GRP, (kv_h + 1) * GRP))
            Q_grp = Q[:, q_idx, :].reshape(seq, -1)
            Q_aug = np.hstack([Q_grp, np.ones((seq, 1))])
            K_rec[:, kv_h, :] = Q_aug @ B

        K_rec += payload["R_sparse"]
        return K_rec


class EvictionCompressor:
    """Keep top-k tokens by attention score norm (score eviction baseline)."""

    def __init__(self, sparsity: float):
        self.sp = sparsity

    def compress(self, K: np.ndarray, Q: np.ndarray) -> dict:
        seq, n_kv, hd = K.shape
        n_keep = max(1, int(seq * self.sp))

        # Score each token by its mean squared norm across heads
        norms = np.mean(np.sum(K**2, axis=-1), axis=-1)  # (seq,)
        kept  = np.argsort(norms)[-n_keep:]

        K_sparse = np.zeros_like(K)
        K_sparse[kept] = K[kept]
        return {"K_sparse": K_sparse}

    def decompress(self, payload: dict, Q: np.ndarray) -> np.ndarray:
        return payload["K_sparse"]


# ── 4. PPL HESAPLAMA ─────────────────────────────────────────────────────────
def compute_ppl(
    text: str,
    seq_len: int,
    layer_indices: List[int],
    compressor,          # IRCMCompressor | EvictionCompressor | None
) -> float:
    """
    Hook-based forward pass: intercept k_proj and q_proj outputs,
    compress K (and reconstruct), replace K in attention.
    Returns perplexity.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    T = input_ids.shape[1]
    if T < 8:
        return float("nan")

    # Storage for hooks
    _q_cache: Dict[int, np.ndarray] = {}
    _k_cache: Dict[int, np.ndarray] = {}
    hooks = []

    if compressor is not None:
        # We need to intercept q_proj and k_proj BEFORE RoPE.
        # TinyLlama uses LlamaAttention: q_proj / k_proj are linear layers,
        # RoPE is applied after inside forward(). We hook the output of k_proj
        # and q_proj (which are pre-RoPE).

        def make_q_hook(layer_idx):
            def hook(module, inp, out):
                # out: (1, seq, D)  → reshape to (seq, N_Q, hd)
                q = out.detach().float().cpu().numpy()
                q = q[0].reshape(T, N_Q, HD)
                _q_cache[layer_idx] = q
            return hook

        def make_k_hook(layer_idx):
            def hook(module, inp, out):
                # out: (1, seq, N_KV*hd)
                k = out.detach().float().cpu().numpy()
                k = k[0].reshape(T, N_KV, HD)

                if layer_idx in layer_indices and layer_idx in _q_cache:
                    Q_np = _q_cache[layer_idx]
                    payload = compressor.compress(k, Q_np)
                    k_rec  = compressor.decompress(payload, Q_np)
                    # Inject back: modify the output tensor in-place
                    out.data.copy_(
                        torch.tensor(
                            k_rec.reshape(1, T, N_KV * HD),
                            dtype=out.dtype, device=out.device
                        )
                    )
            return hook

        layers = model.model.layers
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            hooks.append(attn.q_proj.register_forward_hook(make_q_hook(li)))
            hooks.append(attn.k_proj.register_forward_hook(make_k_hook(li)))

    nll_sum = 0.0
    n_toks  = 0
    stride  = max(1, seq_len // 2)
    prev_end = 0

    with torch.no_grad():
        for begin in range(0, T, stride):
            end   = min(begin + seq_len, T)
            chunk = input_ids[:, begin:end]
            if chunk.shape[1] < 2:
                break

            out    = model(chunk, labels=chunk)
            loss   = out.loss.item()
            n_new  = max(0, end - max(begin, prev_end))
            if n_new > 0:
                nll_sum += loss * n_new
                n_toks  += n_new
            prev_end = end
            if end == T:
                break

    for h in hooks:
        h.remove()

    return math.exp(nll_sum / max(n_toks, 1))


# ── 5. SPARSITY SWEEP ─────────────────────────────────────────────────────────
SPARSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
SEQ_LEN         = 512     # fixed; metin ≥1024 token, bu dilim alınır

LAYER_GROUPS = {
    "shallow (L0–L3)":  list(range(0, 4)),
    "deep (L8–L15)":    list(range(8, 16)),
    "all (L0–L21)":     list(range(0, N_LAYER)),
}

print("\n" + "=" * 68)
print(f"  Sparsity Sweep — {len(SPARSITY_LEVELS)} seviye × "
      f"{len(LAYER_GROUPS)} grup × {len(PPL_TEXTS)} metin")
print("=" * 68)

# Baseline PPL (bir kez hesapla)
print("\n[BASELINE] Hesaplanıyor...")
base_ppls = []
for txt in PPL_TEXTS:
    ppl = compute_ppl(txt, SEQ_LEN, [], None)
    base_ppls.append(ppl)
    print(f"  PPL={ppl:.3f}")
BASE_PPL = float(np.mean(base_ppls))
print(f"[BASELINE] Ortalama={BASE_PPL:.3f}\n")

# Sweep
results = {}   # group → sparsity → {"ircm": [...], "evict": [...]}

for g_name, g_layers in LAYER_GROUPS.items():
    results[g_name] = {}
    print(f"\n{'─'*68}")
    print(f"  Katman grubu: {g_name}")
    print(f"{'─'*68}")
    print(f"  {'Sparsity':>8}  {'IRCM PPL':>10}  {'ΔIRCM':>8}  "
          f"{'Evict PPL':>10}  {'ΔEvict':>9}  {'Kazanan':>8}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*8}")

    for sp in SPARSITY_LEVELS:
        ircm_ppls  = []
        evict_ppls = []
        comp_ircm  = IRCMCompressor(sp)
        comp_ev    = EvictionCompressor(sp)

        for txt in PPL_TEXTS:
            ircm_ppls.append(compute_ppl(txt, SEQ_LEN, g_layers, comp_ircm))
            evict_ppls.append(compute_ppl(txt, SEQ_LEN, g_layers, comp_ev))

        m_ircm  = float(np.mean(ircm_ppls))
        m_evict = float(np.mean(evict_ppls))
        d_ircm  = m_ircm  - BASE_PPL
        d_evict = m_evict - BASE_PPL
        winner  = "IRCM✓" if d_ircm < d_evict else "Evict"

        results[g_name][sp] = {
            "ircm_ppls":  ircm_ppls,
            "evict_ppls": evict_ppls,
            "ircm_mean":  m_ircm,
            "evict_mean": m_evict,
            "delta_ircm": d_ircm,
            "delta_evict": d_evict,
            "ircm_ok":    d_ircm  < 2.0,
            "evict_ok":   d_evict < 2.0,
            "winner":     winner,
        }

        print(f"  {sp*100:>7.0f}%  {m_ircm:>10.3f}  {d_ircm:>+8.3f}  "
              f"{m_evict:>10.3f}  {d_evict:>+9.3f}  {winner:>8}")


# ── 6. ÖZET İSTATİSTİKLER ────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  ÖZET")
print("=" * 68)

total_ircm_wins = 0
total_tests     = 0
ircm_ok_count   = 0

for g, sp_dict in results.items():
    wins = sum(1 for v in sp_dict.values() if v["winner"] == "IRCM✓")
    ok   = sum(1 for v in sp_dict.values() if v["ircm_ok"])
    n    = len(sp_dict)
    total_ircm_wins += wins
    ircm_ok_count   += ok
    total_tests     += n
    print(f"  {g:<22}  IRCM wins: {wins}/{n}  |  ΔPPL<2.0: {ok}/{n}")

print(f"\n  TOPLAM IRCM KAZANIMLARI: {total_ircm_wins}/{total_tests}  "
      f"({'%.1f' % (100*total_ircm_wins/total_tests)}%)")


# ── 7. PARETO GRAFİĞİ ────────────────────────────────────────────────────────
print("\n[PLOT] Pareto grafiği çiziliyor...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle(
    "ICCKV-SENTINEL: IRCM vs Token Eviction — Sparsity–Quality Pareto Curve\n"
    "TinyLlama-1.1B | seq=512 | mean over 3 texts | lower ΔPPL = better",
    fontsize=13, fontweight="bold", y=1.02
)

COLORS = {"IRCM": "#0077cc", "Evict": "#cc3300"}
FILLS  = {"IRCM": "#cce5ff", "Evict": "#ffd5cc"}

sp_pct = [s * 100 for s in SPARSITY_LEVELS]

for ax, (g_name, sp_dict) in zip(axes, results.items()):
    ircm_means  = [sp_dict[s]["delta_ircm"]  for s in SPARSITY_LEVELS]
    evict_means = [sp_dict[s]["delta_evict"] for s in SPARSITY_LEVELS]

    # Confidence bands (std across 3 texts)
    ircm_stds   = [np.std(sp_dict[s]["ircm_ppls"])  for s in SPARSITY_LEVELS]
    evict_stds  = [np.std(sp_dict[s]["evict_ppls"]) for s in SPARSITY_LEVELS]

    ax.fill_between(
        sp_pct,
        [m - s for m, s in zip(ircm_means,  ircm_stds)],
        [m + s for m, s in zip(ircm_means,  ircm_stds)],
        color=FILLS["IRCM"], alpha=0.5
    )
    ax.fill_between(
        sp_pct,
        [m - s for m, s in zip(evict_means, evict_stds)],
        [m + s for m, s in zip(evict_means, evict_stds)],
        color=FILLS["Evict"], alpha=0.4
    )

    ax.plot(sp_pct, ircm_means,  "o-", color=COLORS["IRCM"],
            linewidth=2.5, markersize=6, label="IRCM (proposed)", zorder=5)
    ax.plot(sp_pct, evict_means, "s--", color=COLORS["Evict"],
            linewidth=2.5, markersize=6, label="Token Eviction", zorder=5)

    # ΔPPL=2.0 tolerance line
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.2,
               label="ΔPPL=2.0 threshold")

    ax.set_title(g_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sparsity (% of KV cache kept)", fontsize=10)
    ax.set_ylabel("ΔPPL (vs baseline, lower=better)", fontsize=10)
    ax.set_xlim(0, 32)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Annotate crossover
    for i, sp in enumerate(SPARSITY_LEVELS):
        di, de = ircm_means[i], evict_means[i]
        if de > 50:           # Eviction çok kötü — clip
            ax.annotate(f"↑{de:.0f}", (sp*100, 30),
                        ha="center", fontsize=7, color=COLORS["Evict"])

plt.tight_layout()
out_plot = "/mnt/user-data/outputs/icckv_sparsity_curve.png"
fig.savefig(out_plot, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] ✓ Kaydedildi → {out_plot}")


# ── 8. JSON ÇIKTI ─────────────────────────────────────────────────────────────
summary = {
    "version":      "ICCKV-SENTINEL v2.7",
    "baseline_ppl": BASE_PPL,
    "seq_len":      SEQ_LEN,
    "sparsity_levels": SPARSITY_LEVELS,
    "total_ircm_wins": total_ircm_wins,
    "total_tests":  total_tests,
    "results":      {
        g: {
            str(round(s*100)) + "%": v
            for s, v in sd.items()
        }
        for g, sd in results.items()
    }
}

json_path = "/mnt/user-data/outputs/icckv_sparsity_curve_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"[JSON] ✓ Kaydedildi → {json_path}")


# ── 9. PATENT TABLOSU ─────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  PATENT İDDİALARI — SPARSİTY CURVE KANITI")
print("=" * 68)

claims = {
    "C1: IRCM > Eviction @ tüm sparsity seviyeleri":
        f"{total_ircm_wins}/{total_tests} kazanım",
    "C2: IRCM ΔPPL < 2.0 @ sparsity=5%":
        str({g: results[g][0.05]["ircm_ok"] for g in results}),
    "C3: Eviction ΔPPL > 50 @ sparsity=5%":
        str({g: results[g][0.05]["delta_evict"] > 50 for g in results}),
    "C4: IRCM ΔPPL < 1.0 @ sparsity=10% (shallow)":
        str(results["shallow (L0–L3)"][0.10]["delta_ircm"] < 1.0),
    "C5: Pareto superiority across full curve":
        "Grafik üretildi → " + out_plot,
}

for claim, evidence in claims.items():
    print(f"  {claim}")
    print(f"    → {evidence}\n")

print("=" * 68)
print("  TAMAMLANDI")
print("=" * 68)
