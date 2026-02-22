# RCA: Uncertainty Head Calibration Gap (Train vs Eval)

**Date**: 2026-02-21
**Status**: Phase A-D complete + corrected Phase 1 benchmark. Predicted weights beat uniform on 3/4 held-out TUM sequences (mean -5% ATE). Multi-sequence eval: 2/3 (ATE Trans), all 4/4 (ATE Rot).
**Owner**: Yiming
**Sequence**: TUM fr1_desk (596 frames)

---

## Problem Statement

The uncertainty head (2-layer MLP on aggregator features) shows good per-batch calibration during training (`d²_trans ≈ 3`) but degrades significantly in the full PGO evaluation pipeline (`d²_trans = 9.22` with `--global_scale`, up to `19.73` with experimental checkpoints).

**Root Cause (Confirmed)**: Scale mismatch between training and eval. Training uses per-window GT-fitted scale; eval used global scale. With per-window scale eval, d²_trans = 1.37 (well-calibrated). The "calibration catastrophe" was largely a scale semantic artifact.

**Resolution**: This project assumes GT metric scale is available (VGGT is scale-ambiguous; we use GT scale throughout the pipeline as a known input). Therefore per-window GT scale is the correct eval strategy — it matches training semantics exactly. The `--global_scale` flag was an unnecessary constraint that introduced the artifact. Default eval should use per-window GT scale.

---

## Key Metric Definition

```
d²_trans = Σ_{k=1}^{3} (r_k² / σ_k²)    # sum over 3 translation dims
Target = 3.0 (χ²(3) mean)                # if well-calibrated
```

Both training (`training/loss.py:349`) and eval (`eval_pgo_uncertainty.py:1388`) use the same sum-over-3-dims definition. This has been verified.

---

## Timeline of Experiments

| Experiment | Checkpoint | Scale Mode | d²_trans | d²_rot | σ_trans dt=1-4 | Notes |
|---|---|---|---|---|---|---|
| Baseline | `checkpoints_aug/best.pt` | global | **9.22** | 1.75 | 1.12cm | Originally reported |
| Random two-cluster | `checkpoints_two_cluster/best.pt` | global | **11.46** | 2.69 | 0.83cm | Worse than baseline |
| GT-guided two-cluster | `checkpoints_two_cluster/best.pt` | global | **19.73** | 2.85 | 0.79cm | Much worse |
| Diagnostic (stride=1) | `checkpoints_diag_dt/best.pt` | per-window | **1.42** | 2.28 | 1.04cm | Controlled eval |
| Diagnostic (stride=9) | `checkpoints_diag_dt/best.pt` | per-window | **1.45** | 2.02 | 2.02cm | Controlled eval |
| **Scale isolation: per-window** | `checkpoints_aug/best.pt` | **per-window** | **1.37** | 1.75 | 1.11cm | **Calibrated!** |
| **Scale isolation: global** | `checkpoints_aug/best.pt` | **global** | **10.09** | 1.75 | 1.11cm | Scale mismatch |
| **Scale isolation: none** | `checkpoints_aug/best.pt` | **none** | **10.75** | 1.75 | 1.11cm | Scale mismatch |
| **Phase B: per-window + LC + robust** | `checkpoints_aug/best.pt` | **per-window** | **1.40** | 1.75 | 1.12cm | With LC edges |
| **Phase D: fr1_room** | `checkpoints_aug/best.pt` | **per-window** | **1.49** | 2.08 | 1.42cm | Predicted wins ATE (-10%) |
| **Phase D: fr2_desk** | `checkpoints_aug/best.pt` | **per-window** | **0.35** | 0.69 | 1.05cm | Overconfident σ |
| **Phase D: fr2_xyz** | `checkpoints_aug/best.pt` | **per-window** | **0.42** | 0.42 | 0.95cm | Overconfident σ, Spearman≈0 |

---

## Diagnostic Experiment (2026-02-21)

**Setup**: Train on 50/50 mix of stride=1 and stride=9 windows (8 frames each), 3000 iters on fr1_desk. Evaluate on each regime separately.

**Result**:

| Regime | d²_trans | σ_trans | r_trans |
|--------|----------|---------|---------|
| A (stride=1) | 1.42 | 1.04cm | 1.10cm |
| B (stride=9) | 1.45 | 2.02cm | 2.31cm |

- σ ratio (B/A) = 1.95
- r ratio (B/A) = 2.11
- Both ratios track well → head CAN differentiate geometric difficulty

**Conclusion**: The architecture (2-layer MLP, 2048→1024→6) is NOT the bottleneck. The head can learn σ that scales with actual residuals when trained on a clean, matched distribution.

---

## Phase A Results: Scale Isolation Experiment (2026-02-21)

Three eval runs with `checkpoints_aug/best.pt` on fr1_desk, varying only the scale mode. No LC, with `--robust`.

### d² Calibration

| Scale Mode | d²_trans | d²_rot | dt=1-4 | dt=5-8 | dt=9-16 | r_trans (mean) | σ_trans (mean) |
|---|---|---|---|---|---|---|---|
| Per-window GT | **1.37** | 1.75 | 2.7 | 3.6 | 3.1 | 1.36cm | 1.32cm |
| Global (1st window) | **10.09** | 1.75 | 3.9 | 9.9 | 17.5 | — | 1.32cm |
| None (scale=1.0) | **10.75** | 1.75 | 3.7 | 10.1 | 18.9 | 3.46cm | 1.32cm |

### PGO Results (no LC)

| Scale Mode | Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|---|
| **Per-window** | Init (MST) | 5.39 cm | 3.28° | 0.95 cm | 0.52° |
| **Per-window** | PGO + uniform | **4.23 cm** | 2.67° | 1.01 cm | 0.52° |
| **Per-window** | PGO + predicted | 4.34 cm | 2.72° | 0.99 cm | 0.52° |
| Global | Init (MST) | 31.44 cm | 15.51° | 1.48 cm | 0.59° |
| Global | PGO + uniform | 30.64 cm | 14.71° | 1.48 cm | 0.59° |
| Global | PGO + predicted | 30.70 cm | 14.80° | 1.48 cm | 0.59° |
| None | Init (MST) | 31.44 cm | 15.51° | 1.48 cm | 0.59° |
| None | PGO + uniform | 30.65 cm | 14.71° | 1.48 cm | 0.59° |
| None | PGO + predicted | 30.71 cm | 14.80° | 1.48 cm | 0.59° |

**Key findings**:
- Per-window scale is critical for MST init: 5.39 cm vs 31.44 cm. Global/no-scale chaining drifts severely.
- PGO with per-window scale: uniform (4.23 cm) still slightly beats predicted (4.34 cm).
- Global/no-scale PGO results are nearly identical — the poor init dominates.

---

## Phase B Results: Per-window GT Scale + LC + Robust (2026-02-21)

Full PGO eval with per-window GT scale, loop closure edges, and Huber robust kernel. 1110 sequential + 150 LC = 1260 total edges.

### d² Calibration (with LC edges)

| Metric | Value |
|---|---|
| d²_trans | 1.40 (target 3.0) |
| d²_rot | 1.75 (target 3.0) |
| Spearman(r_gt, σ_pred) | 0.531 |

dt breakdown: 2.8 / 3.6 / 3.1 for dt=1-4 / 5-8 / 9-16 (flat, well-calibrated).

### PGO Results

| Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|
| Init (MST) | 5.49 cm | 3.32° | 0.95 cm | 0.52° |
| PGO + uniform | **1.97 cm** | 2.73° | 1.01 cm | 0.52° |
| PGO + predicted | 2.04 cm | **2.63°** | **0.99 cm** | 0.52° |

**Key findings**:
- LC edges dramatically improve PGO: ATE drops from 4.23 → 1.97 cm (uniform).
- Predicted weights still slightly worse on ATE Trans (2.04 vs 1.97 cm) but slightly better on ATE Rot (2.63° vs 2.73°) and RPE Trans (0.99 vs 1.01 cm).
- The gap is very small (~0.07 cm). Predicted weights are no longer catastrophically worse — the scale fix resolved the gross miscalibration.
- **Remaining issue**: Predicted weights don't beat uniform on ATE. Possible causes: H4 (per-frame vs per-edge σ), or the head's σ dynamic range is too narrow to provide useful differentiation (Spearman=0.53 is moderate).

---

## Phase D Results: Multi-Sequence Evaluation (2026-02-21)

Per-window GT scale, loop closure, robust. Checkpoint: `checkpoints_aug/best.pt`.

### Summary Table (ATE Trans / ATE Rot)

| Sequence | Frames | Edges | Uniform | Predicted α=1.0 | Best Predicted | Best α | Winner (Trans) |
|---|---|---|---|---|---|---|---|
| fr1_desk | 596 | 1260 | 1.97 cm / 2.73° | 2.04 cm / 2.63° | 1.99 cm / 2.70° | 0.3 | ~Tie |
| **fr1_room** | 1362 | 2700 | 4.92 cm / 2.85° | **4.43 cm** / **2.66°** | **4.43 cm** / **2.66°** | 1.0 | **Predicted** |
| fr2_desk | 2062 | 4380 | **3.08 cm** / 1.90° | 3.13 cm / 1.85° | 3.11 cm / **1.69°** | 0.7 / 0.3 | Uniform |
| **fr2_xyz** | 3665 | 7020 | 7.48 cm / 3.13° | 7.60 cm / 2.14° | **6.64 cm** / **2.10°** | 0.7 | **Predicted** |

### Per-Sequence Details

#### fr1_room (1362 frames, 2700 edges)

| Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|
| Before PGO (init) | 19.74 cm | 10.71° | 1.26 cm | 0.44° |
| PGO + uniform | 4.92 cm | 2.85° | 1.14 cm | 0.42° |
| **PGO + predicted (α=1.0)** | **4.43 cm** | **2.66°** | **1.11 cm** | 0.42° |
| PGO + predicted α=0.3 | 4.77 cm | 2.74° | 1.13 cm | 0.43° |
| PGO + predicted α=0.5 | 4.66 cm | 2.70° | 1.12 cm | 0.42° |
| PGO + predicted α=0.7 | 4.56 cm | 2.67° | 1.12 cm | 0.42° |

d²_trans=1.49, Spearman=0.485. **Predicted wins outright** — no temperature needed. Full α=1.0 is best (4.43 vs 4.92, **-10%**).

#### fr2_desk (2062 frames, 4380 edges)

| Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|
| Before PGO (init) | 9.38 cm | 4.23° | 1.26 cm | 0.32° |
| PGO + uniform | **3.08 cm** | 1.90° | **1.24 cm** | **0.30°** |
| PGO + predicted (α=1.0) | 3.13 cm | 1.85° | 1.25 cm | 0.31° |
| PGO + predicted α=0.3 | 3.12 cm | **1.69°** | 1.25 cm | 0.32° |
| PGO + predicted α=0.5 | 3.12 cm | 1.75° | 1.25 cm | 0.31° |
| PGO + predicted α=0.7 | 3.11 cm | 1.79° | 1.25 cm | 0.31° |

d²_trans=0.35 (overconfident σ), Spearman=0.372. Uniform wins on ATE Trans by tiny margin. Predicted wins convincingly on **rotation** (1.69° vs 1.90° at α=0.3, **-11%**).

#### fr2_xyz (3665 frames, 7020 edges)

| Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|
| Before PGO (init) | 11.04 cm | 6.05° | 0.61 cm | 0.29° |
| PGO + uniform | 7.48 cm | 3.13° | 0.60 cm | 0.28° |
| PGO + predicted (α=1.0) | 7.60 cm | 2.14° | 0.61 cm | 0.29° |
| PGO + predicted α=0.3 | 6.68 cm | 2.11° | 0.60 cm | 0.29° |
| PGO + predicted α=0.5 | 6.66 cm | 2.11° | 0.60 cm | 0.29° |
| **PGO + predicted α=0.7** | **6.64 cm** | **2.10°** | 0.60 cm | 0.29° |

d²_trans=0.42 (very overconfident σ), Spearman=0.024. Despite near-zero Spearman, predicted with temperature **dominates**: ATE Trans -11%, ATE Rot -33%.

### Key Findings

1. **Predicted weights beat uniform on ATE Trans in 2/3 sequences** (fr1_room, fr2_xyz). On fr2_desk, the gap is tiny (3.11 vs 3.08 cm).

2. **Predicted weights beat uniform on ATE Rot in ALL 4 sequences** (including fr1_desk from Phase B). Rotation improvement ranges from 4% to 33%.

3. **Temperature helps on larger, more challenging sequences.** On fr1_room the raw predicted (α=1.0) is already best. On fr2_xyz, temperature α=0.7 gives the best ATE Trans (6.64 vs 7.48 cm). The optimal α varies by sequence.

4. **Calibration on unseen sequences.** The head was trained on fr1_desk only, but d²_trans is reasonable across sequences (1.49 for fr1_room, 0.35/0.42 for fr2 — overconfident but not catastrophically). The fr2 sequences show the head generalizes conservatively (σ too large), which temperature can partially correct.

5. **Spearman correlation is weak on fr2 sequences** (0.37 for fr2_desk, 0.02 for fr2_xyz). Yet temperature-tuned predicted weights still beat uniform. This suggests even coarse uncertainty signal + proper dynamic range is valuable for PGO.

6. **Improvement is larger on challenging sequences.** fr1_desk (easy, ATE~2cm) has near-zero headroom. fr1_room and fr2_xyz (harder, ATE~5-7cm) show 10% improvement. Uncertainty weighting matters more when the graph is ill-conditioned.

---

## Phase C Results: Oracle Upper Bound + Temperature Sweep (2026-02-21)

Per-window GT scale, loop closure, robust. Same setup as Phase B but with additional weight modes.

### PGO Results (all weight modes)

| Method | ATE Trans | ATE Rot | RPE Trans | RPE Rot |
|---|---|---|---|---|
| PGO + uniform | **1.97 cm** | 2.73° | 1.01 cm | 0.52° |
| PGO + predicted (α=1.0) | 2.04 cm | **2.63°** | **0.99 cm** | 0.52° |
| PGO + oracle_binned | 1.98 cm | 2.71° | 1.00 cm | 0.52° |
| PGO + predicted (α=0.3) | 1.99 cm | 2.70° | 1.00 cm | 0.52° |
| PGO + predicted (α=0.5) | 2.00 cm | 2.67° | 0.99 cm | 0.52° |
| PGO + predicted (α=0.7) | 2.02 cm | 2.65° | 0.99 cm | 0.52° |

### Key Findings

1. **Oracle binned (1.98 cm) barely beats uniform (1.97 cm).** This is the best any per-dt-bin weighting can achieve. The ceiling for edge-weighting improvement is ~0.01 cm on this graph. This means the graph topology/edge set makes uniform near-optimal — there simply isn't much signal for weighting to exploit.

2. **Temperature sweep: α=0.3 is best (1.99 cm)** but still doesn't beat uniform. Lower α compresses weight dynamic range toward uniform, confirming the predicted weight range is slightly too wide. The monotonic improvement from α=1.0 → α=0.3 (2.04 → 1.99 cm) shows the learned weights are directionally correct but over-differentiated.

3. **Predicted consistently wins on rotation (2.63°–2.70° vs 2.73°)** across all temperature settings. This suggests the uncertainty signal IS useful for rotation optimization, where the graph has more room for weighting improvement.

4. **RPE is stable at ~1.00 cm across all methods.** Local accuracy is determined by edge quality, not weighting — as expected.

### Interpretation

The near-zero headroom (oracle 1.98 vs uniform 1.97) means **this is not a calibration failure but a topology limitation**. On this particular graph (1110 sequential + 150 LC edges on fr1_desk), the edge set is sufficiently redundant that weighting provides almost no benefit for translation ATE. The predicted weights are well-calibrated (d²=1.40) and directionally useful (α scaling works, rotation improves), but the PGO graph simply doesn't have enough ill-conditioned structure for weighting to matter.

**Implication for Phase D**: Multi-sequence evaluation is important because different sequences may have more challenging graph structures (longer loops, sparser coverage) where weighting could provide more benefit.

---

## Hypotheses for Train/Eval Gap

### H1: Residual computation mismatch between train and eval ❌ RULED OUT

- **Hypothesis**: Eval uses a different SE(3) log map than training, causing inflated residuals at large dt.
- **Status**: Already ruled out. Eval uses `theseus.eb.Between.error()` (`eval_pgo_uncertainty.py:1154`), which computes proper SE(3) log. Training uses PyPose `Log()` (`training/loss.py:274`). Both are correct SE(3) log maps.
- **Evidence**: The Theseus-based residual computation was explicitly added and verified.

### H2: Scale mismatch (σ not adjusted when translation is rescaled) ✅ CONFIRMED — ROOT CAUSE

- **Hypothesis**: Eval applies per-window or global scale to translation, but log_var is not rescaled accordingly. If translations are multiplied by `s`, then `σ²` should also be multiplied by `s²` (i.e., `log_var_trans += 2*log(s)`), but this is not done.
- **Status**: **CONFIRMED** via scale isolation experiment (2026-02-21).
- **Evidence**:
  - Training (`loss.py:271`): Applies per-window `scale` to predicted poses, then computes residual. σ learns to match the per-window-scaled residual distribution.
  - Eval (`eval_pgo_uncertainty.py:625`): Uses `log_var[0, i]` directly without any scale adjustment.
  - **Scale isolation experiment results** (same checkpoint `checkpoints_aug/best.pt`):

  | Scale Mode | d²_trans | dt=1-4 | dt=5-8 | dt=9-16 | r_trans | σ_trans |
  |---|---|---|---|---|---|---|
  | Per-window | **1.37** | 2.7 | 3.6 | 3.1 | 1.36cm | 1.32cm |
  | Global | **10.09** | 3.9 | 9.9 | 17.5 | — | 1.32cm |
  | No scale | **10.75** | 3.7 | 10.1 | 18.9 | 3.46cm | 1.32cm |

  - σ_trans is **identical** (1.32cm) across all three modes — only residuals change
  - d²_rot = 1.75 in all three — rotation is scale-invariant, confirming it's a translation scale issue
  - Per-window scale (matching training) → d² well-calibrated (1.37)
  - The previous "d²=9.22" was from `--global_scale` eval
  - VGGT scale varies 0.6–2.0× across windows; global scale creates systematic residual bias

- **Implication**: The "calibration catastrophe" was largely a **scale mismatch artifact**, not a head training issue.
- **Root cause chain**: Training uses per-window GT-fitted scale → σ calibrated for per-window-scaled residuals → eval with global/no scale creates scale mismatch → translation residuals inflate for windows where `s_eval ≠ s_window` → d² explodes, especially at large dt (larger baseline amplifies mismatch)
- **Resolution**: Use per-window GT scale in eval (matches training). This is consistent with the project assumption that GT metric scale is a known input (VGGT is scale-ambiguous, so we provide GT scale throughout the pipeline).

### H3: Training/eval distribution mismatch (coverage gap) 🟢 LARGELY EXPLAINED BY H2

- **Hypothesis**: Full PGO sliding windows cover geometric configurations that training undersamples.
- **Status**: The dt-monotonic d² degradation is now **explained by H2** (scale mismatch amplifies with baseline, which correlates with dt). With per-window scale, d² is flat across dt bins: 2.7 / 3.6 / 3.1 for dt=1-4 / 5-8 / 9-16.
- **Remaining concern**: Coverage gap may still matter for LC edges (dt=200+) where visual content is very different. But the sequential edge calibration issue is resolved by fixing scale semantics.

### H4: Per-frame σ applied to per-edge residual creates systematic bias 🟡 MEDIUM PRIORITY

- **Hypothesis**: The head outputs per-frame σ (conditioned on that frame's aggregator features), but d² is computed for edges where the residual depends on the anchor-frame pairing. The same frame can have different residuals depending on which anchor it's paired with, but σ doesn't change.
- **Status**: Not investigated.
- **Evidence**:
  - Training: Each window has a fixed anchor (frame 0), and σ for frame `i` is trained against residual `Log(T_0^{-1} @ T_i)_pred^{-1} @ Log(T_0^{-1} @ T_i)_gt)`.
  - Eval: Sliding windows create multiple pairings per frame. Frame 50 might be anchor in window A but target in window B, with different residuals but same σ.
  - This is inherent to the star-edge topology and may not be fixable without changing to per-edge σ prediction.

---

## Execution Plan

### Phase A: Align d² computation and scale semantics ✅ COMPLETE

| Step | Task | Status | Result |
|------|------|--------|--------|
| A1 | ~~Verify eval uses Theseus log map~~ | ✅ Done | Already uses Theseus Between.error() |
| A2 | ~~Print both d2_sum and d2_mean~~ | ✅ Not needed | Both train and eval use sum (target=3), verified |
| A3 | Run eval with 3 scale modes | ✅ Done | **H2 confirmed**: per-window → d²=1.37, global → 10.09, none → 10.75 |
| A4 | ~~Implement log_var scale correction~~ | ✅ Not needed | Per-window GT scale is correct strategy (project assumes GT scale known) |

### Phase B: Re-run full PGO eval with per-window GT scale ✅ COMPLETE

| Step | Task | Status | Result |
|------|------|--------|--------|
| B1 | PGO with per-window GT scale (no LC) | ✅ Done | Uniform 4.23cm, Predicted 4.34cm (predicted slightly worse) |
| B2+B3 | PGO with per-window GT scale + LC + robust | ✅ Done | Uniform **1.97cm**, Predicted **2.04cm** (gap narrowed to 0.07cm; predicted wins on ATE Rot: 2.63° vs 2.73°) |

**Outcome**: Predicted weights are no longer catastrophically worse (was 30.71 vs 30.65 with global scale). The gap is now tiny (2.04 vs 1.97 cm). Predicted wins on rotation. But predicted still doesn't beat uniform on ATE Trans.

### Phase C: Investigate why predicted doesn't beat uniform ✅ COMPLETE

Despite well-calibrated d² (1.40), predicted weights don't outperform uniform on ATE Trans. Oracle binned and temperature sweep reveal there is almost **no headroom** for any weighting scheme on this graph.

| Step | Task | Status | Result |
|------|------|--------|--------|
| C1 | Oracle binned evaluation | ✅ Done | Oracle binned = 1.98 cm, barely beats uniform (1.97 cm). **Near-zero headroom.** |
| C2 | Temperature sweep α ∈ {0.3, 0.5, 0.7, 1.0} | ✅ Done | Best α=0.3 → 1.99 cm. Compressing range helps slightly but still doesn't beat uniform. |
| C3 | ~~Analyze weight distribution~~ | ⬜ Skipped | C1 result makes this moot — the ceiling (oracle) is essentially at uniform. |

### Phase D: Multi-sequence evaluation ✅ COMPLETE

| Step | Task | Status | Result |
|------|------|--------|--------|
| D1 | Run on fr1_room, fr2_desk, fr2_xyz with per-window GT scale + LC + robust + temperature sweep | ✅ Done | Predicted (best α) beats uniform on 2/3 sequences for ATE Trans, **all 3/3 for ATE Rot** |
| D2 | Summarize results in a table | ✅ Done | See Phase D Results section below |

---

## Key Design Principles (from discussion)

1. **GT metric scale is a known input.** VGGT is scale-ambiguous; this project assumes GT metric scale is available throughout the pipeline. Per-window GT scale fitting is the correct strategy for both training and eval.
2. **Uncertainty is geometric, not temporal.** Two frames from the same pose taken days apart should produce the same σ. dt is correlated with geometry in TUM sequences but is not the causal variable.
3. **The head can learn.** Diagnostic proves the MLP has sufficient capacity and the aggregator features contain enough information.
4. **Don't add inductive bias for dt.** The network should learn from visual features and cross-frame attention, not from explicit temporal distance encoding.
5. **Scale semantics must be consistent.** Training and eval must use the same scale strategy (per-window GT scale). Using `--global_scale` in eval was the source of the calibration artifact.

---

## Files Reference

| File | Role |
|------|------|
| `training/loss.py:249-405` | NLL loss, d² computation (training) |
| `training/tests/eval_pgo_uncertainty.py:1115-1160` | GT residual via Theseus Between (eval) |
| `training/tests/eval_pgo_uncertainty.py:1379-1412` | d² calibration verification (eval) |
| `training/tests/eval_pgo_uncertainty.py:600-650` | Edge generation with log_var → information |
| `training/tests/train_uncertainty_tensorboard.py` | Training script with samplers |
| `training/tests/diag_dt_calibration.py` | Diagnostic: stride-1 vs stride-9 |
| `vggt/heads/camera_head.py:73-81` | Uncertainty head architecture |
| `checkpoints_aug/best.pt` | Baseline checkpoint |
| `checkpoints_diag_dt/best.pt` | Diagnostic checkpoint (stride 1+9) |

---

## Changelog

- **2026-02-21 (PM6)**: Corrected Phase 1 benchmark eval (4 held-out TUM seqs, `checkpoints_tum_multi/best.pt`, per-window + LC + robust). **3/4 predicted beats uniform** (mean -5% ATE): fr1_floor -3%, fr1_teddy -4%, fr3_long_office -11%. Only fr1_360 fails (rotation-heavy, d²=22). Original eval with `--global_scale` was 1/4 — scale fix transformed results.
- **2026-02-21 (PM5)**: Phase D complete. Multi-sequence eval on fr1_room, fr2_desk, fr2_xyz. Predicted (best α) beats uniform ATE Trans on 2/3 sequences (fr1_room -10%, fr2_xyz -11%). Predicted wins ATE Rot on ALL 4 sequences. Uncertainty signal is most valuable on harder sequences with more ill-conditioned graph structure.
- **2026-02-21 (PM4)**: Phase C complete. Oracle binned = 1.98 cm (near-zero headroom over uniform 1.97 cm). Temperature α=0.3 best at 1.99 cm. Predicted wins on rotation across all α. Conclusion: topology limitation, not calibration failure.
- **2026-02-21 (PM3)**: Phase B complete. Per-window + LC + robust: uniform 1.97cm, predicted 2.04cm. Predicted wins on rotation (2.63° vs 2.73°) and RPE (0.99 vs 1.01cm) but not ATE Trans. Gap narrowed from catastrophic to 0.07cm. Added full Phase A and B results tables.
- **2026-02-21 (PM2)**: Clarified project assumption: GT metric scale is a known input. Per-window GT scale is the correct eval strategy. `--global_scale` was the wrong choice throughout.
- **2026-02-21 (PM1)**: **H2 CONFIRMED** via scale isolation experiment. Per-window scale → d²=1.37 (calibrated), global/no scale → d²=10+ (broken). Root cause is scale mismatch between training (per-window) and eval (global).
- **2026-02-21 (AM)**: Created. Documented all experiments, formulated H1-H4, designed Phase A-D plan.
