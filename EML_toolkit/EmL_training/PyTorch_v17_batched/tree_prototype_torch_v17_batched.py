"""
EML Tree Trainer v17_batched. GPU-batched PyTorch trainer that amortizes
Python / kernel-launch / autograd-graph overhead by running N (seed, strategy)
slots through a single batched EMLTree forward.

Built on v16_final's training methodology with the following additions:
- Batched-seed training: ~14x speedup vs sequential CPU training on RTX 5090
- PerSeedAdam: per-seed Adam optimizer state with surgical reset_seed support
  for clean phase transitions without cross-seed contamination
- Per-seed expression export: snapped tree expression captured for each seed
  to enable failure-mode analysis
- Trajectory history: --save-history captures per-iter snapped expressions
  for trajectory analysis
- Ablation hooks:
  - --lam-active-bypass: penalty on active-path bypass count (regularization)
  - --force-root-right-bypass-from-iter / --until-iter: targeted schedule
    intervention for the root.right gate

Bit-exact reproducible across runs at the metric level (verified across all
800 per-tuple metrics x 18 numeric/integer/boolean fields, see related research
artifacts for full validation). Imports utilities from PyTorch_v16_final/ via
sys.path insertion; depends on v16_final's CUDA support patches.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

V16_DIR = Path(__file__).resolve().parent.parent / "PyTorch_v16_final"
if str(V16_DIR) not in sys.path:
    sys.path.insert(0, str(V16_DIR))
import tree_prototype_torch_v16_final as v16  # noqa: E402

DTYPE = v16.DTYPE
REAL_DTYPE = v16.REAL_DTYPE
_BYPASS_THR = v16._BYPASS_THR
_EML_CLAMP_DEFAULT = v16._EML_CLAMP_DEFAULT


def _init_one_slot(n_leaves: int, n_internal: int, strategy: str, init_scale: float):
    """One (n_leaves, n_internal) init draw — bit-for-bit identical to v16's
    EMLTree.__init__ for the same strategy.

    Caller is responsible for calling torch.manual_seed(seed) before this
    function so the per-slot RNG state matches v16's per-seed sequential init.
    """
    if strategy == "manual":
        leaf_init = torch.zeros(n_leaves, 3, dtype=REAL_DTYPE)
        gate_init = torch.zeros(n_internal, 2, dtype=REAL_DTYPE)
    elif strategy == "biased":
        leaf_init = torch.randn(n_leaves, 3, dtype=REAL_DTYPE) * init_scale
        leaf_init[:, 0] += 2.0
        gate_init = torch.randn(n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
    elif strategy == "uniform":
        leaf_init = torch.randn(n_leaves, 3, dtype=REAL_DTYPE) * init_scale
        gate_init = torch.randn(n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
    elif strategy == "xy_biased":
        leaf_init = torch.randn(n_leaves, 3, dtype=REAL_DTYPE) * init_scale
        leaf_init[:, 1] += 1.0
        leaf_init[:, 2] += 1.0
        gate_init = torch.randn(n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
    elif strategy == "random_hot":
        leaf_init = torch.randn(n_leaves, 3, dtype=REAL_DTYPE) * init_scale
        hot_idx = torch.randint(0, 3, (n_leaves,))
        leaf_init[torch.arange(n_leaves), hot_idx] += 3.0
        gate_init = torch.randn(n_internal, 2, dtype=REAL_DTYPE) * init_scale + 3.0
        open_mask = torch.rand(n_internal, 2) < 0.25
        gate_init[open_mask] -= 6.0
    else:
        leaf_init = torch.randn(n_leaves, 3, dtype=REAL_DTYPE) * init_scale
        gate_init = torch.randn(n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
    return leaf_init, gate_init


class EMLTreeBatched(nn.Module):
    """Full binary tree of depth `depth` with EML at every internal node,
    batched along a leading seed dim of size S.

    Parameters:
        leaf_logits   shape (S, n_leaves, 3)
        blend_logits  shape (S, n_internal, 2)

    The forward path is structurally identical to v16's EMLTree.forward; the
    only changes are an extra leading seed axis on parameters and the
    intermediate `current_level` tensor, plus broadcasting fixes for the
    sigmoid blend coefficients.
    """

    def __init__(
        self,
        depth: int,
        seeds: Sequence[int],
        strategies: Sequence[str],
        init_scale: float = 1.0,
        eml_clamp: float | None = None,
    ):
        super().__init__()
        if len(seeds) != len(strategies):
            raise ValueError(
                f"seeds (len {len(seeds)}) and strategies (len {len(strategies)}) "
                "must have the same length"
            )
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.n_seeds = len(seeds)
        self.eml_clamp = eml_clamp if eml_clamp is not None else _EML_CLAMP_DEFAULT
        self.seeds = list(seeds)
        self.strategies = list(strategies)

        leafs, gates = [], []
        for s, st in zip(seeds, strategies):
            torch.manual_seed(int(s))
            li, gi = _init_one_slot(self.n_leaves, self.n_internal, st, init_scale)
            leafs.append(li)
            gates.append(gi)
        self.leaf_logits = nn.Parameter(torch.stack(leafs, dim=0))
        self.blend_logits = nn.Parameter(torch.stack(gates, dim=0))

    def forward(self, x, y, tau_leaf=1.0, tau_gate=1.0):
        """Batched forward.

        Args:
            x, y: real-domain inputs, shape (batch,). Shared across all seeds.
            tau_leaf, tau_gate: scalar OR per-seed (S,) tensor. Tensor form is
                used when seeds are in different phases (search vs hardening
                with different tau decay schedules). At n_seeds=1 with scalar
                tau, behavior is bit-exact to v16.

        Returns:
            pred:        (S, batch) complex
            leaf_probs:  (S, n_leaves, 3)
            gate_probs:  (S, n_internal, 2)
            eml_outputs: list of (S, batch, n_pairs_at_level) — one per level
        """
        x = x.to(DTYPE)
        y = y.to(DTYPE)
        batch_size = x.shape[0]

        # Broadcast tau across seed dim if tensor; identity if scalar.
        tau_leaf_b = (
            tau_leaf.view(-1, 1, 1)
            if torch.is_tensor(tau_leaf) and tau_leaf.dim() > 0
            else tau_leaf
        )
        tau_gate_b = (
            tau_gate.view(-1, 1, 1)
            if torch.is_tensor(tau_gate) and tau_gate.dim() > 0
            else tau_gate
        )

        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf_b, dim=2)  # (S, n_leaves, 3)
        weights = leaf_probs.to(DTYPE)  # (S, n_leaves, 3)
        ones = torch.ones(batch_size, dtype=DTYPE, device=x.device)
        candidates = torch.stack([ones, x, y], dim=1)  # (batch, 3)
        # candidates (batch, 3) @ weights.T (S, 3, n_leaves) -> (S, batch, n_leaves)
        current_level = torch.matmul(candidates, weights.transpose(-1, -2))

        gate_probs_levels: list[torch.Tensor] = []
        eml_outputs: list[torch.Tensor] = []
        node_idx = 0
        while current_level.shape[2] > 1:
            n_pairs = current_level.shape[2] // 2
            left_children = current_level[:, :, 0::2]   # (S, batch, n_pairs)
            right_children = current_level[:, :, 1::2]  # (S, batch, n_pairs)
            s = torch.sigmoid(
                self.blend_logits[:, node_idx : node_idx + n_pairs] / tau_gate_b
            )  # (S, n_pairs, 2)
            gate_probs_levels.append(s)

            # s_left, s_right: shape (S, 1, n_pairs) — unsqueeze on batch dim
            s_left = s[:, :, 0].unsqueeze(1)
            s_right = s[:, :, 1].unsqueeze(1)
            bypass_left = s_left > _BYPASS_THR
            bypass_right = s_right > _BYPASS_THR
            oml = 1.0 - s_left
            omr = 1.0 - s_right

            lr = torch.where(bypass_left, 1.0, s_left + oml * left_children.real)
            li = torch.where(bypass_left, 0.0, oml * left_children.imag)
            rr = torch.where(bypass_right, 1.0, s_right + omr * right_children.real)
            ri = torch.where(bypass_right, 0.0, omr * right_children.imag)
            left_input = torch.complex(lr, li)
            right_input = torch.complex(rr, ri)

            current_level = v16.eml_exact(left_input, right_input)
            current_level = torch.complex(
                torch.nan_to_num(
                    current_level.real,
                    nan=0.0,
                    posinf=self.eml_clamp,
                    neginf=-self.eml_clamp,
                ).clamp(-self.eml_clamp, self.eml_clamp),
                torch.nan_to_num(
                    current_level.imag,
                    nan=0.0,
                    posinf=self.eml_clamp,
                    neginf=-self.eml_clamp,
                ).clamp(-self.eml_clamp, self.eml_clamp),
            )

            eml_outputs.append(current_level)
            node_idx += n_pairs

        gate_probs = torch.cat(gate_probs_levels, dim=1)  # (S, n_internal, 2)
        return current_level.squeeze(2), leaf_probs, gate_probs, eml_outputs


def _active_path_bypass(gate_probs: torch.Tensor) -> torch.Tensor:
    """Per-seed active-path-weighted bypass count.

    For a binary tree of depth d, each gate's bypass contribution to this
    metric is weighted by the *soft* probability that the gate is on the
    active path — i.e. its output reaches the root given all ancestors'
    bypass states. Dead-branch gates have active-prob near 0 and
    contribute negligibly, eliminating the dilution that ceilinged the
    uniform-mean metric (see bench_reports/FAILURE_MODE_ANALYSIS.md and
    the active-path validation script for derivation and on-baseline
    contrast).

    Trainer-side gate convention:
        gate_probs[..., 0] = sigmoid for the LEFT  side; 1 → bypass
        gate_probs[..., 1] = sigmoid for the RIGHT side; 1 → bypass

    Trainer-side gate ordering (mirrors gateFlatIndex in
    dashboard/lib/expression-tree.ts):
        gates at tree-level l (l=0 is root) span flat indices
        [2**d - 2**(l+1),  2**d - 2**l).

    gate_probs: (S, n_internal, 2)
    Returns:    (S,)
    """
    S, n_internal, _ = gate_probs.shape
    # depth from n_internal: n_internal = 2^d - 1
    depth = (n_internal + 1).bit_length() - 1
    ap = torch.ones(
        S, 1, dtype=gate_probs.dtype, device=gate_probs.device
    )
    total = torch.zeros(S, dtype=gate_probs.dtype, device=gate_probs.device)
    for l in range(depth):
        n_at_level = 1 << l
        gates_below = (1 << depth) - (1 << (l + 1))
        gates_at_level = gate_probs[
            :, gates_below : gates_below + n_at_level, :
        ]  # (S, n_at_level, 2)
        # Bypass contribution: active_prob × gate_prob, summed over both
        # sides. Each gate-side contributes ap × its bypass probability.
        total = total + (ap.unsqueeze(-1) * gates_at_level).sum(dim=(1, 2))
        # Active prob for next level's children: scaled by (1 - bypass).
        if l + 1 < depth:
            left_ap = ap * (1.0 - gates_at_level[:, :, 0])
            right_ap = ap * (1.0 - gates_at_level[:, :, 1])
            # Interleave: child[2p]=left_ap[p], child[2p+1]=right_ap[p]
            ap = torch.stack([left_ap, right_ap], dim=2).reshape(S, -1)
    return total


def compute_losses_batched(
    pred: torch.Tensor,
    target: torch.Tensor,
    leaf_probs: torch.Tensor,
    gate_probs: torch.Tensor,
    eml_outputs: list[torch.Tensor],
    lam_ent,
    lam_bin,
    lam_inter,
    inter_threshold,
    lam_sparse=0.0,
    lam_anti_bypass=0.0,
    lam_active_bypass=0.0,
    uncertainty_power: float = 2.0,
):
    """Per-seed composite objective. Each returned tensor has shape (S,).

    Mirrors v16.compute_losses but reduces along the data-batch axis only,
    keeping the seed axis intact. The scalar to backprop is total.sum()
    (or .mean(); .sum() reproduces v16 single-seed gradients exactly).

    lam_ent, lam_bin, lam_inter, lam_sparse, inter_threshold may be either
    Python scalars or (S,) tensors. Element-wise multiplication with the
    per-seed component tensors gives correct per-seed weighting either way.
    At n_seeds=1 with scalar lambdas this is bit-exact to v16.
    """
    diff = pred - target  # (S, batch) - (batch,) -> (S, batch)
    data_loss = torch.mean(diff.abs() ** 2, dim=1).real  # (S,)
    eps = 1e-12

    leaf_max = leaf_probs.max(dim=2).values  # (S, n_leaves)
    leaf_unc = torch.clamp((1.0 - leaf_max) / (2.0 / 3.0), 0.0, 1.0).pow(uncertainty_power)
    leaf_ent = -(leaf_probs * (leaf_probs + eps).log()).sum(dim=2)  # (S, n_leaves)
    entropy = (leaf_ent * leaf_unc).mean(dim=1)  # (S,)

    gate_unc = torch.clamp(1.0 - (2.0 * gate_probs - 1.0).abs(), 0.0, 1.0).pow(uncertainty_power)
    gate_bin = gate_probs * (1.0 - gate_probs)  # (S, n_internal, 2)
    binarity = (gate_bin * gate_unc).mean(dim=(1, 2))  # (S,)

    sparse = (1.0 - gate_probs).mean(dim=(1, 2))  # (S,)

    S = pred.shape[0]
    inter_penalty = torch.zeros(S, dtype=REAL_DTYPE, device=pred.device)
    # Decide whether to compute inter_penalty at all. Cheap: any seed with
    # lam_inter > 0 means we need it.
    lam_inter_active = (
        (lam_inter > 0).any().item() if torch.is_tensor(lam_inter) else lam_inter > 0
    )
    if lam_inter_active and eml_outputs:
        # Broadcast inter_threshold to (S, 1, 1) when it's a per-seed tensor.
        thr_b = (
            inter_threshold.view(-1, 1, 1)
            if torch.is_tensor(inter_threshold) and inter_threshold.dim() > 0
            else inter_threshold
        )
        for lo in eml_outputs:
            excess = torch.relu(lo.abs() - thr_b)  # (S, batch, n_pairs)
            inter_penalty = inter_penalty + excess.pow(2).mean(dim=(1, 2))
        inter_penalty = inter_penalty / len(eml_outputs)

    total = (
        data_loss
        + lam_ent * entropy
        + lam_bin * binarity
        + lam_inter * inter_penalty
        + lam_sparse * sparse
    )  # (S,)

    # Anti-bypass penalty: per-seed mean gate-probability (high gate_probs
    # = bypassed input). Penalizing this opposes the constant-collapse
    # failure mode — see bench_reports/FAILURE_MODE_ANALYSIS.md. Wrapped
    # in a conditional so that at lam_anti_bypass=0 the autograd graph is
    # bit-identical to the pre-change trainer (regression-safe by
    # construction). Mirrors the lam_inter active-check pattern above.
    lam_anti_bypass_active = (
        (lam_anti_bypass != 0).any().item()
        if torch.is_tensor(lam_anti_bypass)
        else lam_anti_bypass != 0.0
    )
    if lam_anti_bypass_active:
        bypass_share = gate_probs.mean(dim=(1, 2))  # (S,)
        total = total + lam_anti_bypass * bypass_share

    # Active-path bypass: weighted version of the bypass penalty that
    # only counts gates on the active path. See _active_path_bypass for
    # the soft-active-path derivation. Same conditional-wrap pattern as
    # lam_anti_bypass — the autograd graph is structurally identical to
    # pre-change when this is off.
    lam_active_bypass_active = (
        (lam_active_bypass != 0).any().item()
        if torch.is_tensor(lam_active_bypass)
        else lam_active_bypass != 0.0
    )
    if lam_active_bypass_active:
        active_bypass = _active_path_bypass(gate_probs)  # (S,)
        total = total + lam_active_bypass * active_bypass

    leaf_unc_flat = leaf_unc  # (S, n_leaves)
    gate_unc_flat = gate_unc.reshape(S, -1)  # (S, n_internal*2)
    ambiguity = torch.cat([leaf_unc_flat, gate_unc_flat], dim=1).mean(dim=1)  # (S,)

    return total, data_loss, entropy, binarity, inter_penalty, sparse, ambiguity


# ---------------------------------------------------------------------------
# Per-seed helpers (snapshot, restore, evaluate, hard-project, snap analysis)
# ---------------------------------------------------------------------------

def snapshot_seed(tree: EMLTreeBatched, s: int) -> dict:
    """Detached clone of a single seed's parameters."""
    return {
        "leaf_logits": tree.leaf_logits[s].detach().clone(),
        "blend_logits": tree.blend_logits[s].detach().clone(),
    }


def restore_seed(tree: EMLTreeBatched, s: int, snap: dict) -> None:
    """Copy a snapshot back into a single seed slot in place."""
    with torch.no_grad():
        tree.leaf_logits[s].copy_(snap["leaf_logits"])
        tree.blend_logits[s].copy_(snap["blend_logits"])


def evaluate_batched(tree: EMLTreeBatched, x_data, y_data, targets, tau=0.01):
    """Per-seed (S,) MSE / max-real / max-imag. Single CUDA sync at .cpu().

    Mirrors v16.evaluate but returns per-seed vectors instead of scalars.
    """
    with torch.no_grad():
        pred, _, _, _ = tree(x_data, y_data, tau_leaf=tau, tau_gate=tau)
        diff = pred - targets  # (S, batch)
        mse_per = torch.mean(diff.abs() ** 2, dim=1).real  # (S,)
        max_real_per = (pred.real - targets.real).abs().max(dim=1).values  # (S,)
        max_imag_per = pred.imag.abs().max(dim=1).values  # (S,)
    return mse_per, max_real_per, max_imag_per


def hard_project_inplace_batched(tree: EMLTreeBatched, k: float = 24.0) -> None:
    """Snap all seeds' weights to nearest hard 0/1 choice (batched)."""
    with torch.no_grad():
        # leaf: argmax over the 3-choice axis (dim=2), set max -> +k, others -> -k
        lc = torch.argmax(tree.leaf_logits, dim=2)  # (S, n_leaves)
        new_leaf = torch.full_like(tree.leaf_logits, -k)
        new_leaf.scatter_(2, lc.unsqueeze(-1), k)
        tree.leaf_logits.copy_(new_leaf)

        # blend: sign -> ±k
        gc = (tree.blend_logits >= 0).to(tree.blend_logits.dtype)
        new_gate = torch.where(
            gc > 0.5,
            torch.full_like(tree.blend_logits, k),
            torch.full_like(tree.blend_logits, -k),
        )
        tree.blend_logits.copy_(new_gate)


class PerSeedAdam(torch.optim.Optimizer):
    """Adam with independent per-seed state and learning rate.

    Parameters carry a leading seed-batch dim S. Optimizer state per parameter:
        exp_avg      shape (S, ...) — first moment, same shape as parameter
        exp_avg_sq   shape (S, ...) — second moment, same shape as parameter
        step         shape (S,)    — per-seed step counter (independent
                                     bias-correction per slot)

    Per-seed learning rate is set via `set_lr_per_seed(lr_tensor)` and
    consumed by the next `step()` call. If never set, falls back to
    `group['lr']` applied uniformly.

    Surgical reset for phase transitions / NaN restarts: `reset_seed(slot)`
    zeros exp_avg, exp_avg_sq, and step for one slot — no other slots
    affected. This matches v16's behavior of `optimizer = torch.optim.Adam(
    tree.parameters(), lr=args.lr)` at phase transition (which fully reset
    state for the only-running seed).

    At S=1 this reduces to PyTorch's standard Adam mathematically. The op
    sequence differs slightly (broadcasting tensor lr/bc1/bc2 vs scalar in
    PyTorch's `addcdiv_`); empirically the per-iter divergence stays at the
    float64 ulp scale through full hardening (verified by trajectory gate).
    """

    def __init__(self, params, n_seeds: int, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.n_seeds = int(n_seeds)
        self._lr_per_seed: torch.Tensor | None = None

    def set_lr_per_seed(self, lr_tensor: torch.Tensor) -> None:
        """Set per-seed lr for the next step(). Tensor shape: (S,)."""
        if lr_tensor.shape != (self.n_seeds,):
            raise ValueError(
                f"expected lr shape ({self.n_seeds},), got {tuple(lr_tensor.shape)}"
            )
        self._lr_per_seed = lr_tensor

    def reset_seed(self, slot: int) -> None:
        """Zero exp_avg, exp_avg_sq, and step for a single seed slot.

        Use this after `restore_seed(tree, slot, state)` so the slot starts
        from the restored params with fresh Adam state — matching v16's
        `tree.load_state_dict(...); optimizer = Adam(...)` pattern at phase
        transition / NaN restart.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is None or len(state) == 0:
                    continue
                state["exp_avg"][slot].zero_()
                state["exp_avg_sq"][slot].zero_()
                state["step"][slot] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        S = self.n_seeds
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr_default = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros(S, dtype=p.dtype, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                # Per-seed step increment
                step_t.add_(1)

                # First and second moment updates (in-place, per-element).
                # Identical to PyTorch's _single_tensor_adam ops.
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Per-seed bias corrections (S,)
                bc1 = 1.0 - beta1 ** step_t
                bc2 = 1.0 - beta2 ** step_t
                bc2_sqrt = bc2.sqrt()

                # Per-seed lr (use group default if not set externally)
                if self._lr_per_seed is not None:
                    lr_per = self._lr_per_seed
                else:
                    lr_per = torch.full((S,), lr_default, dtype=p.dtype, device=p.device)

                step_size = lr_per / bc1  # (S,)

                # Broadcast (S,) -> (S, 1, 1, ...) for elementwise ops on (S, ...) tensor
                view_shape = (S,) + (1,) * (p.dim() - 1)
                bc2_sqrt_b = bc2_sqrt.view(view_shape)
                step_size_b = step_size.view(view_shape)

                # PyTorch ref: denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                denom = exp_avg_sq.sqrt().div_(bc2_sqrt_b).add_(eps)

                # PyTorch ref: param.addcdiv_(exp_avg, denom, value=-step_size)
                # We can't use addcdiv_ with a tensor `value`. Equivalent op:
                #   p -= step_size_b * (exp_avg / denom)
                # Decomposed for in-place efficiency:
                update = exp_avg / denom
                p.addcmul_(update, step_size_b, value=-1.0)

        return loss


def reset_optimizer_slot(optimizer, tree: EMLTreeBatched, slot: int) -> None:
    """Compatibility wrapper. Prefer `optimizer.reset_seed(slot)` directly
    when using PerSeedAdam. Kept for any code paths still constructing
    `torch.optim.Adam` (which has per-tensor not per-slot state)."""
    if isinstance(optimizer, PerSeedAdam):
        optimizer.reset_seed(slot)
        return
    for p in tree.parameters():
        st = optimizer.state.get(p, None)
        if st is None:
            continue
        if "exp_avg" in st:
            st["exp_avg"][slot].zero_()
        if "exp_avg_sq" in st:
            st["exp_avg_sq"][slot].zero_()


def clip_grad_norm_per_seed(tree: EMLTreeBatched, max_norm: float = 1.0, eps: float = 1e-6):
    """Per-seed L2 gradient norm clip. Each of the S seeds is clipped
    independently — no cross-seed contamination through a shared global norm.

    Mirrors `torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)`
    but with the per-seed leading dimension preserved: instead of one scalar
    `total_norm` over all parameters, we compute `(S,)` per-seed norms by
    summing squared gradients over all non-seed dims of each parameter.

    At n_seeds=1 the computation reduces to the same numerical result as
    `clip_grad_norm_`, modulo float-summation order (verified bit-exact in
    practice for the v17 parameter shapes).

    Returns the per-seed gradient norms before clipping (S,) for diagnostics.
    """
    S = tree.n_seeds
    sq_sum = torch.zeros(
        S, dtype=tree.leaf_logits.dtype, device=tree.leaf_logits.device
    )
    for p in tree.parameters():
        if p.grad is None:
            continue
        sq = (p.grad.detach() ** 2).reshape(S, -1).sum(dim=1)  # (S,)
        sq_sum = sq_sum + sq
    norm_per = torch.sqrt(sq_sum)  # (S,)
    clip_coef = (max_norm / (norm_per + eps)).clamp(max=1.0)  # (S,)
    for p in tree.parameters():
        if p.grad is None:
            continue
        view = clip_coef.view(-1, *([1] * (p.grad.dim() - 1)))
        p.grad.mul_(view)
    return norm_per


_LEAF_NAMES = ("1", "x", "y")


def extract_expression_batched(
    tree: EMLTreeBatched, slot: int, snap_threshold: float = 0.01
) -> dict:
    """Per-seed JSON-serializable snapshot of the tree topology.

    Reads the *continuous* tree's softmax/sigmoid distributions and labels
    each entry confident/uncertain by the same threshold rule as
    `analyze_snap_batched`. Confident entries emit only the discrete choice
    (smaller payload); uncertain entries also emit the soft probabilities so
    the dashboard can render "60% x / 40% y" style annotations.

    Output schema (depth-3 tree shown for shape):
        {
          "depth": 3,
          "leaves": [
            {"choice": "x", "confident": true},
            {"choice": null, "confident": false, "probs": [0.10, 0.55, 0.35]},
            ...  # length = 2**depth
          ],
          "gates": [
            {"left": 1, "right": 0, "confident": true},
            {"left": 0.42, "right": 0.91, "confident": false},
            ...  # length = 2**depth - 1
          ],
        }
    """
    with torch.no_grad():
        leaf_p = torch.softmax(tree.leaf_logits[slot], dim=1).cpu().numpy()  # (n_leaves, 3)
        gate_p = torch.sigmoid(tree.blend_logits[slot]).cpu().numpy()        # (n_internal, 2)

    leaves = []
    for i in range(tree.n_leaves):
        row = leaf_p[i]
        max_p = float(row.max())
        max_idx = int(row.argmax())
        if max_p >= 1.0 - snap_threshold:
            leaves.append({"choice": _LEAF_NAMES[max_idx], "confident": True})
        else:
            leaves.append({
                "choice": None,
                "confident": False,
                "probs": [float(row[0]), float(row[1]), float(row[2])],
            })

    gates = []
    for i in range(tree.n_internal):
        l, r = float(gate_p[i, 0]), float(gate_p[i, 1])
        l_conf = (l <= snap_threshold) or (l >= 1.0 - snap_threshold)
        r_conf = (r <= snap_threshold) or (r >= 1.0 - snap_threshold)
        if l_conf and r_conf:
            gates.append({"left": int(l > 0.5), "right": int(r > 0.5), "confident": True})
        else:
            gates.append({"left": l, "right": r, "confident": False})

    return {"depth": tree.depth, "leaves": leaves, "gates": gates}


def analyze_snap_batched(tree: EMLTreeBatched, snap_threshold: float = 0.01):
    """Per-seed uncertainty analysis. Returns list of dicts of length S."""
    with torch.no_grad():
        leaf_probs = torch.softmax(tree.leaf_logits, dim=2).cpu().numpy()  # (S, n_leaves, 3)
        gate_probs = torch.sigmoid(tree.blend_logits).cpu().numpy()  # (S, n_internal, 2)

    names = {0: "1", 1: "x", 2: "y"}
    out = []
    for s in range(tree.n_seeds):
        uncertain_leaves = []
        for i in range(tree.n_leaves):
            max_p = leaf_probs[s, i].max()
            max_idx = leaf_probs[s, i].argmax()
            if max_p < 1.0 - snap_threshold:
                uncertain_leaves.append(
                    f"leaf[{i}]: best={names[max_idx]}({max_p:.4f}) "
                    f"probs=[{leaf_probs[s, i, 0]:.4f}, "
                    f"{leaf_probs[s, i, 1]:.4f}, {leaf_probs[s, i, 2]:.4f}]"
                )
        uncertain_gates = []
        for i in range(tree.n_internal):
            for j in range(2):
                p = gate_probs[s, i, j]
                if snap_threshold < p < 1.0 - snap_threshold:
                    side = "left" if j == 0 else "right"
                    uncertain_gates.append(f"gate[{i}].{side}: prob={p:.4f}")
        out.append({
            "uncertain_leaves": uncertain_leaves,
            "uncertain_gates": uncertain_gates,
            "n_uncertain": len(uncertain_leaves) + len(uncertain_gates),
        })
    return out


# ---------------------------------------------------------------------------
# Batched training loop (step 1 of the deferred-work move sequence)
# ---------------------------------------------------------------------------

def train_seed_batch(
    args,
    seed_strategy_pairs: list[tuple[int, str]],
    x_train,
    y_train,
    t_train,
    manual_init_fn=None,
    log_prefix: str = "",
    record_history: bool = False,
):
    """Train one batched run covering S = len(seed_strategy_pairs) slots.

    Returns: (tree, snapped_tree, summaries_per_seed). Each summary mirrors
    the dict that v16.train_one_seed would have returned for that single
    (seed, strategy).

    Design notes (step 1 scope):
      - Per-seed phase (search / hardening): each slot tracks its own phase
        independently; the search->hardening transition fires per slot when
        plateau-or-budget conditions are met (matches v16 semantics).
      - Per-seed tau, lam_ent, lam_bin: built each iter as (S,) tensors and
        broadcast through the forward + loss. At n_seeds=1 this collapses to
        scalar behavior (bit-exact validated).
      - Per-seed lr_mult: applied via gradient scaling before optimizer.step().
        Approximate equivalence to v16's lr scaling — Adam state evolves with
        scaled gradients, which causes a small momentum deviation vs v16. Step
        2 of the deferred work (per-seed grad clip) and step 3 (per-seed Adam
        restart) refine this further.
      - Convergence mask: when a slot's hard-RMSE drops below success_thr for
        early_stop_count consecutive evals, snapshot its params, mark it
        converged, and zero its gradients on subsequent steps. The whole batch
        terminates when >=95% of slots are converged or have hit the iter
        budget.
      - Aggregated sync: one .cpu() per iter to pull (per-seed total loss,
        per-seed data loss, finite mask) to host. Eval-step extras come at
        eval_every cadence.
      - Step 1 simplifications: NO NaN-restart logic (any iter with non-finite
        total loss is skipped wholesale), NO per-seed grad-clip, NO L-BFGS
        polish, NO Mathematica/pt export.
    """
    import math
    from copy import deepcopy

    device = args.device
    S = len(seed_strategy_pairs)
    seeds = [int(s) for s, _ in seed_strategy_pairs]
    strategies = [st for _, st in seed_strategy_pairs]

    # Build batched tree on CPU first so manual init (which writes CPU
    # tensors into leaf/blend) doesn't hit a device mismatch.
    tree = EMLTreeBatched(
        depth=args.depth,
        seeds=seeds,
        strategies=strategies,
        init_scale=args.init_scale,
        eml_clamp=args.eml_clamp,
    )

    # Manual init applied per slot (each slot may have its own seed for noise)
    if manual_init_fn is not None:
        for s_idx, (seed, strategy) in enumerate(seed_strategy_pairs):
            if strategy != "manual":
                continue
            single = v16.EMLTree(
                depth=args.depth,
                init_scale=args.init_scale,
                init_strategy="biased",
                eml_clamp=args.eml_clamp,
            )
            manual_init_fn(single)
            if args.init_noise > 0:
                v16.add_init_noise(single, args.init_noise, seed=seed)
            with torch.no_grad():
                tree.leaf_logits[s_idx].copy_(single.leaf_logits)
                tree.blend_logits[s_idx].copy_(single.blend_logits)

    tree = tree.to(device)
    optimizer = PerSeedAdam(tree.parameters(), n_seeds=S, lr=args.lr)

    # Per-seed CPU-side state (small and updated once per iter at most).
    INF = float("inf")
    best_soft_loss = [INF] * S
    best_soft_state: list[dict | None] = [None] * S
    best_hard_loss = [INF] * S
    best_hard_state: list[dict | None] = [None] * S
    plateau_counter = [0] * S
    hard_success_streak = [0] * S
    hard_trigger_streak = [0] * S
    nan_skip_count = [0] * S  # current streak — resets after restart
    nan_total_events = [0] * S  # lifetime NaN iter count — never resets
    nan_restart_count = [0] * S  # lifetime restart count — never resets

    phase_per = ["search"] * S
    hardening_iter_per: list[int | None] = [None] * S
    hard_step_per = [0] * S

    converged = [False] * S

    # Per-seed history (when record_history=True). Lists of length-T arrays.
    # Each "iter row" is appended to the running stacks below.
    histories: list[dict | None] = (
        [
            {
                "iter": [],
                "soft_rmse": [],
                "best_soft_rmse": [],
                "tau": [],
                "entropy": [],
                "binarity": [],
                "eval_iter": [],
                "hard_rmse": [],
                # Per-eval snapped expression: one entry per `eval_iter`
                # appended below. Used for trajectory inspection (e.g.
                # detecting when a near-miss seed first reaches and then
                # abandons the canonical form).
                "expression": [],
                "snap_eval_iter": [],
                "snap_rmse": [],
            }
            for _ in range(S)
        ]
        if record_history
        else [None] * S
    )
    converged_at_iter: list[int | None] = [None] * S

    total_iters = args.search_iters + args.hardening_iters
    n_evals = 0

    print(f"{log_prefix}batch start: S={S} total_iters={total_iters} device={device}")
    for it in range(1, total_iters + 1):
        # ---- batch termination check ----
        n_done = sum(converged)
        if S > 0 and n_done >= max(1, int(0.95 * S)) and n_done < S:
            # If 95% converged but not all, keep going only if remaining slots
            # are still actively progressing. Strict 95% halt.
            print(f"{log_prefix}it={it} early-terminate batch ({n_done}/{S} converged >= 95%)")
            break
        if n_done == S:
            print(f"{log_prefix}it={it} all {S} seeds converged")
            break

        # ---- per-seed phase transition (search -> hardening) ----
        for s_idx in range(S):
            if converged[s_idx]:
                continue
            if phase_per[s_idx] == "search":
                if it > args.search_iters or (
                    plateau_counter[s_idx] >= args.patience
                    and best_soft_loss[s_idx] < args.patience_threshold
                ):
                    phase_per[s_idx] = "hardening"
                    hardening_iter_per[s_idx] = it
                    hard_step_per[s_idx] = 0
                    if best_soft_state[s_idx] is not None:
                        restore_seed(tree, s_idx, best_soft_state[s_idx])
                        # **CRITICAL**: also reset this slot's Adam state so
                        # the search-phase momentum doesn't corrupt the first
                        # hardening update. Mirrors v16's
                        # `tree.load_state_dict(best); optimizer = Adam(...)`
                        # at phase transition.
                        optimizer.reset_seed(s_idx)

        # ---- build per-seed tau, lam_ent, lam_bin, lr_mult ----
        tau_list = [0.0] * S
        lam_ent_list = [0.0] * S
        lam_bin_list = [0.0] * S
        lr_mult_list = [0.0] * S
        any_active = False
        for s_idx in range(S):
            if converged[s_idx]:
                # Frozen — gradients will be zeroed; tau/lambdas don't matter
                # for backward, but tau still affects forward (and therefore
                # the gate_probs/leaf_probs feeding loss for OTHER seeds isn't
                # affected — each seed's branch is independent). Use any
                # finite values.
                tau_list[s_idx] = args.tau_hard
                lam_ent_list[s_idx] = args.lam_ent_hard
                lam_bin_list[s_idx] = args.lam_bin_hard
                lr_mult_list[s_idx] = 0.0
                continue
            any_active = True
            if phase_per[s_idx] == "search":
                tau_list[s_idx] = args.tau_search
                lam_ent_list[s_idx] = 0.0
                lam_bin_list[s_idx] = 0.0
                lr_mult_list[s_idx] = 1.0
            else:
                hs = hard_step_per[s_idx]
                if hs >= args.hardening_iters:
                    # Hardening complete for this slot — treat as converged.
                    converged[s_idx] = True
                    converged_at_iter[s_idx] = it
                    if best_hard_state[s_idx] is not None:
                        restore_seed(tree, s_idx, best_hard_state[s_idx])
                    elif best_soft_state[s_idx] is not None:
                        restore_seed(tree, s_idx, best_soft_state[s_idx])
                    tau_list[s_idx] = args.tau_hard
                    lam_ent_list[s_idx] = args.lam_ent_hard
                    lam_bin_list[s_idx] = args.lam_bin_hard
                    lr_mult_list[s_idx] = 0.0
                    continue
                t = hs / max(1, args.hardening_iters)
                t_tau = t ** args.hardening_tau_power
                tau_list[s_idx] = args.tau_search * (args.tau_hard / args.tau_search) ** t_tau
                lam_ent_list[s_idx] = t * args.lam_ent_hard
                lam_bin_list[s_idx] = t * args.lam_bin_hard
                lr_mult_list[s_idx] = max(args.hardening_lr_floor, (1.0 - t) ** 2)
                hard_step_per[s_idx] = hs + 1

        if not any_active:
            print(f"{log_prefix}it={it} no active seeds — exit")
            break

        tau_t = torch.tensor(tau_list, dtype=REAL_DTYPE, device=device)
        lam_ent_t = torch.tensor(lam_ent_list, dtype=REAL_DTYPE, device=device)
        lam_bin_t = torch.tensor(lam_bin_list, dtype=REAL_DTYPE, device=device)
        lr_mult_t = torch.tensor(lr_mult_list, dtype=REAL_DTYPE, device=device)
        active_t = torch.tensor(
            [0.0 if c else 1.0 for c in converged],
            dtype=REAL_DTYPE,
            device=device,
        )

        # ---- forward + loss ----
        optimizer.zero_grad()
        pred, lp, gp, eml_outs = tree(x_train, y_train, tau_leaf=tau_t, tau_gate=tau_t)
        total, data_loss, entropy, binarity, _, _, _ = compute_losses_batched(
            pred, t_train, lp, gp, eml_outs,
            lam_ent_t, lam_bin_t, args.lam_inter, args.inter_threshold,
            lam_sparse=0.0, lam_anti_bypass=args.lam_anti_bypass,
            lam_active_bypass=args.lam_active_bypass,
            uncertainty_power=1.0,
        )

        # ---- finite check (one sync) ----
        # Per-seed NaN handling: when some seeds produce non-finite total
        # (NaN or Inf), we DO NOT skip the iter — we mask those seeds out of
        # the backward chain so the other (finite) seeds train normally.
        finite_per = torch.isfinite(total)
        all_finite = bool(finite_per.all().item())

        # ---- backward ----
        if all_finite:
            loss_for_backward = total.sum()
        else:
            # Replace non-finite total entries with detached zeros so they
            # contribute zero loss. The backward chain through the NaN-
            # producing seeds' forward graph is short-circuited by the
            # `where`: dL/d(total[NaN]) = 0, but PyTorch may still propagate
            # NaN through the chain (0 * NaN = NaN in IEEE-754). To handle
            # this we ALSO sanitize gradients post-backward via nan_to_num.
            finite_mask_no_grad = finite_per.detach()
            total_safe = torch.where(
                finite_mask_no_grad, total, torch.zeros_like(total)
            )
            loss_for_backward = total_safe.sum()
        loss_for_backward.backward()

        # Defensive: zero out any NaN/Inf gradients that snuck through. This
        # protects the optimizer (and Adam state) from corruption regardless
        # of how the NaN propagated through autograd.
        if not all_finite:
            with torch.no_grad():
                for p in tree.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-seed L2 norm clip — no cross-seed contamination through a shared
        # global norm. At n_seeds=1 reduces to the same result as v16's
        # global clip_grad_norm_.
        clip_grad_norm_per_seed(tree, max_norm=1.0)

        # ---- per-seed lr via PerSeedAdam.set_lr_per_seed ----
        # Mirrors v16's `optimizer.param_groups[0]["lr"] = args.lr * lr_mult`
        # but per-seed: each slot gets `args.lr * lr_mult_per_seed`. Adam's
        # m and v accumulate raw gradients (no scaling pollution); the
        # per-seed lr only scales the final update step.
        # Converged + NaN seeds get lr=0 so their params don't move at all.
        with torch.no_grad():
            lr_per = (args.lr * lr_mult_t * active_t).clone()
            if not all_finite:
                lr_per.mul_(finite_per.to(REAL_DTYPE))
        optimizer.set_lr_per_seed(lr_per)

        optimizer.step()

        # Schedule intervention (Ablation 4): force the root's right
        # gate to bypass starting at a configured iter, optionally
        # ending at another configured iter. The window is half-open:
        # [from_iter, until_iter). When until_iter is None (default),
        # the override applies indefinitely from from_iter onward
        # (Ablation 4 used this form). Both flags default to None so
        # baseline runs are unchanged — the conditional is fully gated
        # off in baseline/ablation-1/2/3 runs.
        #
        # The 1-iter form `--from N --until N+1` applies the override
        # for exactly one iter, used by Ablation 5b to bound the
        # "shock duration matters" question from below.
        if (
            getattr(args, "force_root_right_bypass_from_iter", None) is not None
            and it >= args.force_root_right_bypass_from_iter
            and (
                getattr(args, "force_root_right_bypass_until_iter", None) is None
                or it < args.force_root_right_bypass_until_iter
            )
        ):
            with torch.no_grad():
                # Root gate is the last in the flat blend_logits array
                # (n_internal - 1 = 2^depth - 2). Right side is column 1.
                root_idx = tree.n_internal - 1
                tree.blend_logits[:, root_idx, 1].fill_(args.init_k)

        # ---- per-seed NaN-restart bookkeeping ----
        if not all_finite:
            with torch.no_grad():
                nan_per_cpu = (~finite_per).cpu().tolist()
            for s_idx, n in enumerate(nan_per_cpu):
                if converged[s_idx]:
                    continue
                if n:
                    nan_skip_count[s_idx] += 1
                    nan_total_events[s_idx] += 1
                    plateau_counter[s_idx] += 1
                    nan_streak = nan_skip_count[s_idx]
                    # Hit restart threshold? Restore last best-soft state and
                    # reset Adam momentum for this slot. Then track restarts.
                    if nan_streak >= args.nan_restart_patience:
                        if best_soft_state[s_idx] is not None:
                            restore_seed(tree, s_idx, best_soft_state[s_idx])
                        reset_optimizer_slot(optimizer, tree, s_idx)
                        nan_skip_count[s_idx] = 0  # streak resets
                        nan_restart_count[s_idx] += 1  # lifetime restart count
                        if it % args.eval_every == 0 or s_idx == 0:
                            print(
                                f"{log_prefix}  seed={seeds[s_idx]} "
                                f"({strategies[s_idx]}) it={it} NAN-RESTART "
                                f"#{nan_restart_count[s_idx]} "
                                f"(restored best-soft + reset Adam)"
                            )
            if it % args.eval_every == 0:
                n_nan = sum(nan_per_cpu)
                print(f"{log_prefix}it={it} non-finite seeds: {n_nan} (continuing for the {S - n_nan} healthy seeds)")

        # ---- aggregated sync: pull per-seed scalars to CPU once ----
        with torch.no_grad():
            stack = torch.stack([data_loss, entropy, binarity], dim=0).detach()  # (3, S)
            stack_cpu = stack.cpu().numpy()
        data_loss_cpu = stack_cpu[0]
        entropy_cpu = stack_cpu[1]
        binarity_cpu = stack_cpu[2]

        # ---- per-seed best-soft tracking ----
        for s_idx in range(S):
            if converged[s_idx]:
                continue
            soft_loss = float(data_loss_cpu[s_idx])
            if math.isfinite(soft_loss) and soft_loss < best_soft_loss[s_idx]:
                rel_imp = (best_soft_loss[s_idx] - soft_loss) / max(best_soft_loss[s_idx], 1e-15)
                best_soft_loss[s_idx] = soft_loss
                best_soft_state[s_idx] = snapshot_seed(tree, s_idx)
                plateau_counter[s_idx] = (
                    0 if rel_imp > args.plateau_rtol else plateau_counter[s_idx] + 1
                )
            else:
                plateau_counter[s_idx] += 1

        # ---- per-seed history (only if requested) ----
        if record_history:
            for s_idx in range(S):
                if converged[s_idx]:
                    continue
                soft_loss = float(data_loss_cpu[s_idx])
                h = histories[s_idx]
                h["iter"].append(it)
                h["soft_rmse"].append(math.sqrt(max(soft_loss, 0.0)) if math.isfinite(soft_loss) else float("nan"))
                bsl = best_soft_loss[s_idx]
                h["best_soft_rmse"].append(
                    math.sqrt(max(bsl, 0.0)) if math.isfinite(bsl) else float("nan")
                )
                h["tau"].append(float(tau_list[s_idx]))
                h["entropy"].append(float(entropy_cpu[s_idx]))
                h["binarity"].append(float(binarity_cpu[s_idx]))

        # ---- periodic eval at tau_hard ----
        do_eval = (it % max(1, args.eval_every) == 0)
        if any(
            phase_per[i] == "hardening" and tau_list[i] <= args.tail_eval_tau
            for i in range(S)
        ):
            do_eval = do_eval or (it % max(1, args.tail_eval_every) == 0)

        if do_eval:
            n_evals += 1
            mse_per, _, _ = evaluate_batched(
                tree, x_train, y_train, t_train, tau=args.tau_hard
            )
            hard_mse_cpu = mse_per.detach().cpu().numpy()

            for s_idx in range(S):
                if converged[s_idx]:
                    continue
                hm = float(hard_mse_cpu[s_idx])
                if math.isfinite(hm) and hm < best_hard_loss[s_idx]:
                    best_hard_loss[s_idx] = hm
                    best_hard_state[s_idx] = snapshot_seed(tree, s_idx)
                if record_history:
                    h = histories[s_idx]
                    h["eval_iter"].append(it)
                    h["hard_rmse"].append(
                        math.sqrt(max(hm, 0.0)) if math.isfinite(hm) else float("nan")
                    )
                    # Capture the seed's snapped expression at this eval
                    # point for trajectory inspection. Adds ~1ms per
                    # seed per eval (CPU work, no GPU sync penalty
                    # beyond the existing eval-batched sync).
                    h["expression"].append(
                        extract_expression_batched(
                            tree, s_idx, args.snap_threshold
                        )
                    )
                if phase_per[s_idx] == "hardening" and math.isfinite(hm) and hm < args.success_thr:
                    hard_success_streak[s_idx] += 1
                else:
                    hard_success_streak[s_idx] = 0
                if phase_per[s_idx] == "search" and math.isfinite(hm) and hm < args.hard_trigger_mse:
                    hard_trigger_streak[s_idx] += 1
                elif phase_per[s_idx] == "search":
                    hard_trigger_streak[s_idx] = 0

                if hard_success_streak[s_idx] >= args.early_stop_count:
                    converged[s_idx] = True
                    converged_at_iter[s_idx] = it
                    # Snapshot best-hard state at convergence for later restore
                    if best_hard_state[s_idx] is None:
                        best_hard_state[s_idx] = snapshot_seed(tree, s_idx)
                    print(
                        f"{log_prefix}  seed={seeds[s_idx]} ({strategies[s_idx]}) "
                        f"it={it} CONVERGED  best_hard_mse={best_hard_loss[s_idx]:.3e}"
                    )
                    continue

                if (
                    phase_per[s_idx] == "search"
                    and hard_trigger_streak[s_idx] >= args.hard_trigger_count
                ):
                    phase_per[s_idx] = "hardening"
                    hardening_iter_per[s_idx] = it + 1
                    hard_step_per[s_idx] = 0
                    if best_soft_state[s_idx] is not None:
                        restore_seed(tree, s_idx, best_soft_state[s_idx])
                        optimizer.reset_seed(s_idx)
                    hard_trigger_streak[s_idx] = 0

            # progress print
            n_active = S - sum(converged)
            n_search = sum(1 for i, p in enumerate(phase_per) if p == "search" and not converged[i])
            n_hard = sum(1 for i, p in enumerate(phase_per) if p == "hardening" and not converged[i])
            print(
                f"{log_prefix}it={it:6d}  active={n_active:3d}/{S}  "
                f"min_soft={min(best_soft_loss):.3e}  "
                f"min_hard={min(best_hard_loss):.3e}  "
                f"phases=[s={n_search}, h={n_hard}, c={sum(converged)}]"
            )

    # ---- finalize: restore best-hard (or best-soft) per seed ----
    with torch.no_grad():
        for s_idx in range(S):
            if best_hard_state[s_idx] is not None:
                restore_seed(tree, s_idx, best_hard_state[s_idx])
            elif best_soft_state[s_idx] is not None:
                restore_seed(tree, s_idx, best_soft_state[s_idx])

    snapped_tree = deepcopy(tree)
    hard_project_inplace_batched(snapped_tree)
    snap_mse_per, snap_max_real_per, snap_max_imag_per = evaluate_batched(
        snapped_tree, x_train, y_train, t_train, tau=0.01
    )
    snap_mse_cpu = snap_mse_per.detach().cpu().numpy()
    snap_max_real_cpu = snap_max_real_per.detach().cpu().numpy()
    snap_max_imag_cpu = snap_max_imag_per.detach().cpu().numpy()

    snap_infos = analyze_snap_batched(tree, args.snap_threshold)

    summaries = []
    for s_idx in range(S):
        snap_mse = float(snap_mse_cpu[s_idx])
        snap_rmse = math.sqrt(max(snap_mse, 0.0)) if math.isfinite(snap_mse) else float("nan")
        fit_success = bool(math.isfinite(snap_mse) and snap_mse < args.fit_success_thr)
        symbol_success = bool(math.isfinite(snap_mse) and snap_mse < args.success_thr)
        stable = bool(symbol_success and snap_infos[s_idx]["n_uncertain"] <= args.max_uncertain_success)
        summary = {
            "seed": seeds[s_idx],
            "strategy": strategies[s_idx],
            "snap_mse": snap_mse,
            "snap_rmse": snap_rmse,
            "snap_max_real": float(snap_max_real_cpu[s_idx]),
            "snap_max_imag": float(snap_max_imag_cpu[s_idx]),
            "fit_success": fit_success,
            "symbol_success": symbol_success,
            "stable_symbol_success": stable,
            "success": fit_success,
            "n_uncertain": snap_infos[s_idx]["n_uncertain"],
            "hardening_iter": hardening_iter_per[s_idx],
            "nan_skips": nan_skip_count[s_idx],  # streak at end of training (resets on restart)
            "nan_total_events": nan_total_events[s_idx],  # lifetime NaN iter count
            "nan_restarts": nan_restart_count[s_idx],  # lifetime restart count
            "converged_during_train": converged[s_idx],
            "converged_at_iter": converged_at_iter[s_idx],
            "expression": extract_expression_batched(tree, s_idx, args.snap_threshold),
        }
        if record_history and histories[s_idx] is not None:
            h = histories[s_idx]
            # Append final snap RMSE row to match v16's plot-friendly hist format
            snap_eval_x = (h["iter"][-1] + 1) if h["iter"] else 1
            h["snap_eval_iter"].append(snap_eval_x)
            h["snap_rmse"].append(snap_rmse)
            summary["history"] = h
        summaries.append(summary)

    return tree, snapped_tree, summaries


# ---------------------------------------------------------------------------
# Per-seed export helpers (slice batched tensors to single-seed v16.EMLTree
# instances so we can reuse v16's known-good export machinery).
# ---------------------------------------------------------------------------

def make_single_seed_tree(batched_tree: EMLTreeBatched, slot: int, eml_clamp=None):
    """Return a v16.EMLTree whose parameters match the given slot of the
    batched tree. CPU tensors (export code is CPU-only)."""
    single = v16.EMLTree(
        depth=batched_tree.depth, init_strategy="biased",
        eml_clamp=eml_clamp if eml_clamp is not None else batched_tree.eml_clamp,
    )
    with torch.no_grad():
        single.leaf_logits.copy_(batched_tree.leaf_logits[slot].detach().cpu())
        single.blend_logits.copy_(batched_tree.blend_logits[slot].detach().cpu())
    return single


def export_seed_artifacts(
    tree: EMLTreeBatched,
    snapped_tree: EMLTreeBatched,
    slot: int,
    seed_stem: str,
    args,
) -> None:
    """Write `.pt`, `.m` (snapped), `_continuous.m` for a single slot.
    Mirrors v16.main()'s per-run artifact layout."""
    single_continuous = make_single_seed_tree(tree, slot, eml_clamp=args.eml_clamp)
    single_snapped = make_single_seed_tree(snapped_tree, slot, eml_clamp=args.eml_clamp)
    torch.save(single_continuous.state_dict(), f"{seed_stem}.pt")
    single_snapped.export_mathematica(
        f"{seed_stem}.m", discretize=True, snap_threshold=args.snap_threshold
    )
    single_continuous.export_mathematica(
        f"{seed_stem}_continuous.m", discretize=False
    )


# ---------------------------------------------------------------------------
# CLI (mirrors v16 + adds --n-seeds / --batch-seeds)
# ---------------------------------------------------------------------------

def parse_args_batched():
    import argparse
    p = argparse.ArgumentParser(
        description="EML tree trainer v17_batched — seed-axis batched"
    )
    a = p.add_argument

    # Target / shape (same as v16)
    a("--target-fn", type=str, default="eml_depth3",
      choices=sorted(v16.TARGET_FUNCTIONS.keys()))
    a("--depth", type=int, default=3)
    a("--init-scale", type=float, default=1.0)
    a("--init-strategy", type=str, default="all",
      help="biased/uniform/xy_biased/random_hot/manual/all")
    a("--init-expr", type=str, default="")
    a("--init-blend", type=str, default="")
    a("--init-leaves", type=str, default="")
    a("--init-k", type=float, default=32.0)
    a("--init-noise", type=float, default=0.0)

    # Seeds — v17 semantics
    a("--seed0", type=int, default=137)
    a("--n-seeds", type=int, default=1,
      help="number of distinct seed values (campaign size). Default 1.")
    a("--seeds", type=int, default=None,
      help="alias for --n-seeds (v16 backward compatibility)")
    a("--batch-seeds", type=int, default=None,
      help="number of (seed, strategy) tuples to run concurrently per batched "
           "forward. Default = total tuples = n_seeds * n_strategies "
           "(single batch covers entire campaign).")

    # Data
    a("--data-lo", type=float, default=1.0)
    a("--data-hi", type=float, default=3.0)
    a("--data-step", type=float, default=0.1)
    a("--gen-lo", type=float, default=0.5)
    a("--gen-hi", type=float, default=5.0)
    a("--generalization-points", type=int, default=4000)

    # Optimization (same as v16)
    a("--search-iters", type=int, default=6000)
    a("--hardening-iters", type=int, default=2000)
    a("--lr", type=float, default=0.01)
    a("--tau-search", type=float, default=2.5)
    a("--tau-hard", type=float, default=0.01)
    a("--hardening-tau-power", type=float, default=2.0)
    a("--hardening-lr-floor", type=float, default=0.01)
    a("--patience", type=int, default=4200)
    a("--patience-threshold", type=float, default=1e-2)
    a("--plateau-rtol", type=float, default=1e-3)
    a("--lam-ent-hard", type=float, default=2e-2)
    a("--lam-bin-hard", type=float, default=2e-2)
    a("--lam-inter", type=float, default=1e-4)
    a("--inter-threshold", type=float, default=50.0)
    # Anti-bypass: penalty on per-seed mean gate-probability (high gate_probs
    # = bypassed input). Default 0.0 preserves existing baseline behavior;
    # set positive to penalize the constant-collapse failure mode.
    # See bench_reports/FAILURE_MODE_ANALYSIS.md.
    a("--lam-anti-bypass", type=float, default=0.0)
    # Active-path-bypass: weighted version of anti-bypass that only counts
    # gates on the active path (each gate's bypass contribution scaled by
    # the soft probability that its output reaches the root). On the d3
    # baseline this metric has a 0.73 contrast between canonical and
    # cc-to-0 populations vs the uniform-mean's 0.006 — see
    # _validate_active_path_metric.py for the calibration.
    a("--lam-active-bypass", type=float, default=0.0)
    # Schedule intervention (Ablation 4): force the root's right gate
    # to bypass starting at a configured iter. Default None preserves
    # existing behavior (gate evolves freely under gradient). When set,
    # at every iter ≥ value, override blend_logits[:, root, 1] = init_k
    # (saturated sigmoid → bypass). Tests whether forcing the gate
    # near-miss seeds are stuck on flips them to canonical.
    a("--force-root-right-bypass-from-iter", type=int, default=None)
    # Optional upper bound for the override window. Default None means
    # "apply from from_iter onward indefinitely" (Ablation 4 form).
    # When set, override applies in the half-open window
    # [from_iter, until_iter). Used by Ablation 5 to test shock-duration
    # sensitivity (300-iter vs 1-iter override windows).
    a("--force-root-right-bypass-until-iter", type=int, default=None)
    a("--eml-clamp", type=float, default=1e300)

    # Diagnostics
    a("--eval-every", type=int, default=200)
    a("--tail-eval-every", type=int, default=50)
    a("--tail-eval-tau", type=float, default=0.2)
    a("--early-stop-count", type=int, default=10)
    a("--hard-trigger-mse", type=float, default=1e-20)
    a("--hard-trigger-count", type=int, default=3)
    a("--nan-restart-patience", type=int, default=50)
    a("--max-nan-restarts", type=int, default=100)

    # Success
    a("--fit-success-thr", type=float, default=1e-6)
    a("--success-thr", type=float, default=1e-20)
    a("--snap-threshold", type=float, default=0.01)
    a("--max-uncertain-success", type=int, default=0)

    # Polish (deferred — accepted for CLI parity but ignored)
    a("--lbfgs-steps", type=int, default=0)
    a("--lbfgs-lr", type=float, default=0.6)

    # Device
    a("--device", type=str, default="auto",
      help="cpu, cuda, cuda:0, or auto")

    # Output
    a("--save-prefix", type=str, default="v17_run")
    a("--export-m", type=str, default="eml_tree_v17_batched.m")
    a("--skip-plot", action="store_true")
    a("--no-history", action="store_true",
      help="disable per-seed history recording (skips per-seed loss plots)")
    a("--save-history", action="store_true",
      help="retain per-seed history in the metrics JSON. Default strips it "
           "to keep file size reasonable. Enable for trajectory analysis "
           "(e.g. mid-training expression snapshots for near-miss inspection).")
    a("--loss-y-min", type=float, default=1e-16)
    a("--loss-y-max", type=float, default=1e1)
    a("--plot-dpi", type=int, default=300)
    a("--plot-title-fontsize", type=float, default=13.0)
    a("--plot-label-fontsize", type=float, default=15.0)
    a("--plot-tick-fontsize", type=float, default=12.0)
    a("--plot-legend-fontsize", type=float, default=13.0)
    a("--plot-title", type=str, default="")

    args = p.parse_args()

    if args.depth < 0:
        raise ValueError("--depth must be >= 0")
    if args.init_strategy not in ["biased", "uniform", "xy_biased", "random_hot", "manual", "all"]:
        raise ValueError(f"Unsupported --init-strategy: {args.init_strategy}")

    # Resolve --seeds alias for --n-seeds
    if args.seeds is not None and args.n_seeds == 1:
        args.n_seeds = args.seeds

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device} requested but CUDA is not available")

    return args


def main_batched():
    """v17 batched campaign entry point."""
    import argparse
    import json
    import math
    from copy import deepcopy
    from datetime import datetime

    args = parse_args_batched()
    target_fn, target_desc = v16.get_target_fn(args.target_fn)

    # Strategies
    if args.init_strategy == "all":
        strategies = ["biased", "uniform", "xy_biased", "random_hot"]
    else:
        strategies = [args.init_strategy]

    # Manual init handling — mirror v16's logic (init_expr can override depth)
    manual_init_fn = v16.make_manual_init_fn(args)
    if manual_init_fn is not None:
        strategies = ["manual"]
        if args.init_noise == 0:
            seeds = [args.seed0]
        else:
            seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    elif args.init_strategy == "manual":
        raise ValueError(
            "--init-strategy=manual requires --init-expr or --init-blend/--init-leaves"
        )
    else:
        seeds = list(range(args.seed0, args.seed0 + args.n_seeds))

    run_plan: list[tuple[int, str]] = [(s, st) for s in seeds for st in strategies]
    n_total = len(run_plan)

    # Resolve batch size
    if args.batch_seeds is None or args.batch_seeds <= 0:
        args.batch_seeds = n_total
    args.batch_seeds = min(args.batch_seeds, n_total)

    # Output dir
    output_dir = Path(args.save_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now()
    stamp = run_start.strftime("%Y%m%d-%H%M%S")
    run_tag = output_dir.name

    # Tee stdout to log file (matches v16)
    stdout_log = output_dir / f"{run_tag}_stdout_{stamp}.log"
    log_handle = open(stdout_log, "w", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    sys.stdout = v16.TeeStream(orig_stdout, log_handle)

    try:
        print(f"Command: {' '.join(sys.argv)}")
        print(f"Run start: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir.resolve()}")

        # Data
        x_train, y_train, t_train = v16.make_grid_data(
            target_fn, lo=args.data_lo, hi=args.data_hi, step=args.data_step
        )
        x_gen, y_gen, t_gen = v16.make_generalization_data(
            target_fn, lo=args.gen_lo, hi=args.gen_hi,
            n=args.generalization_points,
        )
        x_train = x_train.to(args.device)
        y_train = y_train.to(args.device)
        t_train = t_train.to(args.device)
        x_gen = x_gen.to(args.device)
        y_gen = y_gen.to(args.device)
        t_gen = t_gen.to(args.device)

        device_label = args.device
        if args.device.startswith("cuda"):
            device_label = f"{args.device} ({torch.cuda.get_device_name(0)})"

        n_batches = math.ceil(n_total / args.batch_seeds)

        print()
        print("=== EML Tree Training v17_batched ===")
        print(f"device: {device_label}")
        print(f"target: {args.target_fn} = {target_desc}")
        print(f"depth={args.depth}  leaves={2**args.depth}  internal={2**args.depth - 1}")
        print(f"campaign: {len(seeds)} seeds * {len(strategies)} strategies = {n_total} runs")
        print(f"batch-seeds={args.batch_seeds}  -> {n_batches} batched run(s)")
        print(f"search_iters={args.search_iters}  hardening_iters={args.hardening_iters}")
        print()

        all_summaries: list[dict] = []
        record_history = not args.no_history

        for batch_idx in range(n_batches):
            chunk = run_plan[batch_idx * args.batch_seeds : (batch_idx + 1) * args.batch_seeds]
            batch_tag = f"batch{batch_idx + 1:03d}of{n_batches:03d}"
            print(f"--- {batch_tag}: S={len(chunk)} ---")
            tree, snapped_tree, summaries = train_seed_batch(
                args, chunk, x_train, y_train, t_train,
                manual_init_fn=manual_init_fn if any(st == "manual" for _, st in chunk) else None,
                log_prefix="  ",
                record_history=record_history,
            )
            # Per-seed exports
            for slot, summary in enumerate(summaries):
                seed = summary["seed"]
                strategy = summary["strategy"]
                run_idx_global = batch_idx * args.batch_seeds + slot + 1
                seed_stem = (
                    f"{run_tag}_run{run_idx_global:03d}_seed{seed}_{strategy}_{stamp}"
                )
                seed_base = output_dir / seed_stem
                export_seed_artifacts(tree, snapped_tree, slot, str(seed_base), args)

                # Generalization eval (per-seed, single forward)
                with torch.no_grad():
                    pred_g, _, _, _ = tree(x_gen, y_gen, tau_leaf=0.01, tau_gate=0.01)
                    diff_g = pred_g[slot] - t_gen
                    gen_mse = float((diff_g.abs() ** 2).mean().real.item())
                    gen_max_real = float((pred_g[slot].real - t_gen.real).abs().max().item())
                    gen_max_imag = float(pred_g[slot].imag.abs().max().item())
                summary["gen_mse"] = gen_mse
                summary["gen_max_real"] = gen_max_real
                summary["gen_max_imag"] = gen_max_imag

                # Per-seed loss plot
                if record_history and not args.skip_plot and "history" in summary:
                    plot_path = png_dir / f"{seed_stem}_loss.png"
                    plot_title = (
                        args.plot_title if args.plot_title
                        else f"EML v17_batched | {batch_tag} slot{slot} seed={seed} ({strategy})"
                    )
                    try:
                        v16.save_loss_plot(
                            str(plot_path),
                            summary["history"],
                            title=plot_title,
                            args=args,
                            hardening_iter=summary["hardening_iter"],
                        )
                    except Exception as ex:
                        print(f"  WARN: loss plot for seed {seed} failed: {ex}")

                # Strip history from summary before persisting (large by
                # default — per-seed × per-eval entries balloon JSON size).
                # Pass --save-history to retain it for trajectory analysis
                # (e.g. inspecting when near-miss seeds first reach
                # canonical mid-training).
                if not args.save_history:
                    summary.pop("history", None)
                all_summaries.append(summary)

        # Aggregate run-level metrics
        n_success = sum(1 for s in all_summaries if s["success"])
        n_fit = sum(1 for s in all_summaries if s["fit_success"])
        n_sym = sum(1 for s in all_summaries if s["symbol_success"])
        n_stable = sum(1 for s in all_summaries if s["stable_symbol_success"])
        rmses = sorted(s["snap_rmse"] for s in all_summaries)

        print()
        print("=" * 60)
        print(
            f"Results: success={n_success}/{n_total}  fit={n_fit}/{n_total}  "
            f"symbol={n_sym}/{n_total}  stable_symbol={n_stable}/{n_total}"
        )
        print(
            f"snap_rmse  min={rmses[0]:.3e}  median={rmses[len(rmses) // 2]:.3e}  "
            f"max={rmses[-1]:.3e}"
        )

        # Run-level metrics JSON
        metrics = {
            "version": "v17_batched",
            "target_fn": args.target_fn,
            "target_desc": target_desc,
            "depth": args.depth,
            "device": args.device,
            "n_seeds": len(seeds),
            "n_strategies": len(strategies),
            "n_total": n_total,
            "batch_seeds": args.batch_seeds,
            "n_batches": n_batches,
            "search_iters": args.search_iters,
            "hardening_iters": args.hardening_iters,
            "n_success": n_success,
            "n_fit_success": n_fit,
            "n_symbol_success": n_sym,
            "n_stable_symbol_success": n_stable,
            "runs": all_summaries,
        }
        metrics_path = output_dir / f"{run_tag}_metrics_{stamp}.json"
        # Sanitize NaN/Inf so json.dump(allow_nan=False) is valid
        sanitized = v16._sanitize_for_json(metrics)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(sanitized, fh, indent=2, allow_nan=False)
        print(f"Saved metrics: {metrics_path}")

        run_end = datetime.now()
        print(f"Run end: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {run_end - run_start}")
    finally:
        sys.stdout = orig_stdout
        log_handle.close()


__all__ = [
    "EMLTreeBatched",
    "compute_losses_batched",
    "_init_one_slot",
    "snapshot_seed",
    "restore_seed",
    "evaluate_batched",
    "hard_project_inplace_batched",
    "analyze_snap_batched",
    "extract_expression_batched",
    "clip_grad_norm_per_seed",
    "PerSeedAdam",
    "reset_optimizer_slot",
    "train_seed_batch",
    "make_single_seed_tree",
    "export_seed_artifacts",
    "parse_args_batched",
    "main_batched",
]


if __name__ == "__main__":
    main_batched()
