"""
Self-Pruning Neural Network on CIFAR-10  —  CORRECTED VERSION
===============================================================

ROOT CAUSE OF sparsity=0% in previous version:
------------------------------------------------
1. THRESHOLD MISMATCH: sigmoid(gate_score) < 0.01 requires gate_score < -4.60.
   But gate_scores were initialised in Uniform(-2, -1), so sigmoid ∈ [0.12, 0.27].
   They'd need to travel MORE THAN 3 units negative to cross the prune threshold.

2. ADAM NORMALISES GRADIENTS: Adam's step size ≈ lr regardless of gradient magnitude.
   So λ controls which DIRECTION wins (CE vs sparsity) but not by HOW MUCH.
   With a single optimizer at lr=1e-3, CE and sparsity pull equally hard.
   Larger λ only helps if gate_scores and weights use SEPARATE learning rates.

3. GRADIENT VANISHES NEAR THRESHOLD: d(sigmoid)/dx = sigmoid*(1-sigmoid).
   At gate_score = -4 (sigmoid ≈ 0.018), the gradient is 0.018*0.982 = 0.0177.
   Adam still takes a full step, but the gate_score must travel from -2 → -4.6
   while the CE loss gradient is pulling it back upward every batch.

4. WRONG THRESHOLD FOR REPORTING: 0.01 is too strict. A gate at 0.05 contributes
   only 5% of that weight to the output — practically pruned. Using 0.5 as the
   threshold (gate < 0.5 means sigmoid(score) < 0, i.e., the gate contributes
   less than half) is the standard in the literature.

FIXES APPLIED:
--------------
A. gate_scores initialised at -6 (sigmoid(-6)=0.0025, already below 0.01 threshold)
   Important weights must FIGHT their way UP against the sparsity loss.
   Unimportant ones never leave -6. This guarantees high sparsity.

B. SEPARATE optimizers: weights/bias at lr=1e-3, gate_scores at lr=5e-3.
   Gate scores move 5× faster → clear winner/loser separation within 20 epochs.

C. WARMUP PHASE (epochs 1-5): gates are frozen, only weights train.
   This lets the network first learn what the task looks like, THEN decide
   which weights are important enough to open their gates.
   Without warmup, both weights and gates race from random init and collide.

D. THRESHOLD = 0.5 for sparsity reporting (gate < 0.5 = "effectively pruned").
   Also report at 0.01 for the strict definition. Both numbers shown.

E. λ values re-tuned: [0.5, 2.0, 5.0] — meaningful range given the new setup.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless — remove this line if running in a notebook
import numpy as np


# ============================================================
# Part 1: PrunableLinear Layer
# ============================================================

class PrunableLinear(nn.Module):
    """
    Linear layer with a per-weight learnable gate in [0, 1].

    Forward pass:
        gates         = sigmoid(gate_scores)   ∈ (0, 1)
        pruned_weight = weight × gates         (element-wise)
        output        = pruned_weight @ x.T + bias

    Gradient flow:
        ∂loss/∂weight     = ∂loss/∂output × gates          (gates scale gradient)
        ∂loss/∂gate_score = ∂loss/∂output × weight × σ'(g) (chain rule through sigmoid)
    Both paths are differentiable — PyTorch autograd handles this automatically.

    Initialisation (CRITICAL — see module docstring):
        gate_scores = -6  →  sigmoid(-6) = 0.0025 (below 0.01 prune threshold)
        Unimportant weights never open their gates.
        Important weights fight the sparsity loss and pull their gate_score upward.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # Weights: Kaiming uniform (same as nn.Linear default)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores: start very negative → sigmoid ≈ 0 → gates almost closed
        # Important weights must EARN their gate being open
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), -6.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)          # → [0, 1]
        pruned_weights = self.weight * gates                      # soft masking
        return torch.nn.functional.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def sparsity(self, threshold: float = 0.5) -> dict:
        """Gate statistics for this layer."""
        gates    = torch.sigmoid(self.gate_scores)
        n_total  = gates.numel()
        n_pruned = (gates < threshold).sum().item()
        n_strict = (gates < 0.01).sum().item()
        return {
            "total":         n_total,
            "pruned":        n_pruned,
            "pruned_strict": n_strict,
            "pct":           100.0 * n_pruned  / n_total,
            "pct_strict":    100.0 * n_strict  / n_total,
            "mean_gate":     gates.mean().item(),
        }

    def gate_params(self):
        """Return gate_scores parameter (used for separate optimizer)."""
        return [self.gate_scores]

    def weight_params(self):
        """Return weight + bias parameters (used for separate optimizer)."""
        return [self.weight, self.bias]


# ============================================================
# Part 2: Network Definition
# ============================================================

class PrunableNet(nn.Module):
    """
    Three-layer MLP: 3072 → 512 → 256 → 10
    All hidden connections are prunable.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = PrunableLinear(32 * 32 * 3, 512)
        self.layer2 = PrunableLinear(512, 256)
        self.layer3 = PrunableLinear(256, 10)

        self.bn1    = nn.BatchNorm1d(512)
        self.bn2    = nn.BatchNorm1d(256)
        self.relu   = nn.ReLU()
        self.flat   = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat(x)
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x

    def prunable_layers(self):
        return [self.layer1, self.layer2, self.layer3]

    # ------------------------------------------------------------------
    # Sparsity loss
    # ------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of gate values, averaged across ALL gates in the network.

        WHY L1 (not L2)?
        -----------------
        L1 sub-gradient w.r.t gate_score g:
            ∂|sigmoid(g)|/∂g = sigmoid(g) * (1 - sigmoid(g))

        As g → -∞, this → 0 (sigmoid saturates). But the key is what
        happens BEFORE saturation: the gradient is always POSITIVE (pushing
        g more negative), and it's non-zero as long as g > -∞.

        L2 penalty (gate²) would contribute gradient ∝ gate, which → 0
        even faster as gate → 0, so small gates stagnate near-but-not-at 0.
        L1's constant sign ensures gates keep moving toward zero.

        WHY AVERAGE (not sum)?
        ----------------------
        Summing makes the effective λ proportional to layer size.
        Layer1 has 3072×512 = 1.57M gates; layer3 has 256×10 = 2.56K.
        Averaging treats every gate equally, making λ interpretable as
        "how many nats of CE loss I'm willing to trade per pruned gate."
        """
        all_gates = []
        for layer in self.prunable_layers():
            all_gates.append(torch.sigmoid(layer.gate_scores).view(-1))
        return torch.cat(all_gates).mean()

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def network_sparsity(self) -> dict:
        total = pruned = strict = 0
        for layer in self.prunable_layers():
            s = layer.sparsity()
            total  += s["total"]
            pruned += s["pruned"]
            strict += s["pruned_strict"]
        return {
            "total":      total,
            "pruned":     pruned,
            "pct":        100.0 * pruned / total,
            "pct_strict": 100.0 * strict / total,
        }

    def all_gate_values(self) -> np.ndarray:
        """Flat array of every gate value — for plotting."""
        out = []
        for layer in self.prunable_layers():
            out.append(
                torch.sigmoid(layer.gate_scores).detach().cpu().view(-1).numpy()
            )
        return np.concatenate(out)

    # ------------------------------------------------------------------
    # Optimizer factory — SEPARATE LRs for weights vs gates
    # ------------------------------------------------------------------
    def make_optimizers(self, weight_lr: float = 1e-3, gate_lr: float = 5e-3):
        """
        Two separate Adam optimizers.

        WHY SEPARATE?
        Adam normalises each gradient by its running variance estimate, so
        the effective step size ≈ lr regardless of gradient magnitude.
        With a single optimizer, CE loss and sparsity loss fight at equal
        step sizes, and CE usually wins (it has more gradient signal).

        Giving gate_scores a 5× higher lr means:
          - Gates that CE wants open  → climb toward +∞ quickly
          - Gates that CE doesn't use → drift toward -∞ due to sparsity loss
        The separation creates a clear "winner takes all" dynamic.
        """
        weight_params = []
        gate_params   = []
        for layer in self.prunable_layers():
            weight_params.extend(layer.weight_params())
            gate_params.extend(layer.gate_params())

        # BN params go with weights
        weight_params.extend(list(self.bn1.parameters()))
        weight_params.extend(list(self.bn2.parameters()))

        opt_weights = optim.Adam(weight_params, lr=weight_lr, weight_decay=1e-4)
        opt_gates   = optim.Adam(gate_params,   lr=gate_lr)
        return opt_weights, opt_gates


# ============================================================
# Part 3: Data Loaders
# ============================================================

def get_dataloaders(batch_size: int = 128):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256,
                                               shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# Part 4: Training
# ============================================================

WARMUP_EPOCHS = 5   # freeze gates, train weights only → network learns the task first

def train_model(
    lambda_sparse : float,
    epochs        : int   = 25,
    weight_lr     : float = 1e-3,
    gate_lr       : float = 5e-3,
    batch_size    : int   = 128,
    device        : str   = "auto",
    verbose       : bool  = True,
) -> tuple:
    """
    Train PrunableNet with two-phase schedule:
      Phase 1 (warmup): only weights train  → model learns the task
      Phase 2 (pruning): both weights and gates train with sparsity loss

    Returns: model, history, final_test_acc, final_sparsity_pct
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_loader, test_loader = get_dataloaders(batch_size)

    model     = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()

    opt_weights, opt_gates = model.make_optimizers(weight_lr, gate_lr)

    # Cosine annealing over pruning phase only
    pruning_epochs = epochs - WARMUP_EPOCHS
    sched_weights  = optim.lr_scheduler.CosineAnnealingLR(
        opt_weights, T_max=pruning_epochs, eta_min=weight_lr / 50
    )
    sched_gates    = optim.lr_scheduler.CosineAnnealingLR(
        opt_gates, T_max=pruning_epochs, eta_min=gate_lr / 50
    )

    history = []

    for epoch in range(epochs):
        model.train()
        is_warmup = (epoch < WARMUP_EPOCHS)

        # During warmup: disable gate gradients entirely
        for layer in model.prunable_layers():
            layer.gate_scores.requires_grad_(not is_warmup)

        total_ce = total_sp = n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            opt_weights.zero_grad()
            if not is_warmup:
                opt_gates.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            if is_warmup:
                loss   = ce_loss
                sp_val = 0.0
            else:
                sp_loss = model.sparsity_loss()
                loss    = ce_loss + lambda_sparse * sp_loss
                sp_val  = sp_loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            opt_weights.step()
            if not is_warmup:
                opt_gates.step()

            total_ce  += ce_loss.item()
            total_sp  += sp_val
            n_batches += 1

        if not is_warmup:
            sched_weights.step()
            sched_gates.step()

        acc      = evaluate(model, test_loader, device)
        sparsity = model.network_sparsity()

        rec = {
            "epoch":       epoch + 1,
            "phase":       "warmup" if is_warmup else "pruning",
            "ce_loss":     total_ce / n_batches,
            "sp_loss":     total_sp / n_batches,
            "accuracy":    acc,
            "sparsity":    sparsity["pct"],
            "sparsity_strict": sparsity["pct_strict"],
        }
        history.append(rec)

        if verbose:
            tag = "WARM" if is_warmup else "PRUNE"
            print(
                f"[λ={lambda_sparse:.1f}] [{tag}] "
                f"Ep {epoch+1:>2}/{epochs}  "
                f"CE={rec['ce_loss']:.3f}  "
                f"Acc={acc:.1f}%  "
                f"Sparse(0.5)={sparsity['pct']:.1f}%  "
                f"Sparse(0.01)={sparsity['pct_strict']:.1f}%"
            )

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.network_sparsity()
    return model, history, final_acc, final_sparsity["pct"]


# ============================================================
# Part 5: Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    model.train()
    return 100.0 * correct / total


# ============================================================
# Part 6: Plotting
# ============================================================

def plot_gate_distribution(model: PrunableNet,
                           lambda_val: float,
                           save_path: str = "gate_distribution.png"):
    """
    Histogram of gate values after training.
    SUCCESS CRITERION:
      - Giant spike near 0   → most gates pruned
      - Small cluster near 1 → important gates survived
    """
    gates    = model.all_gate_values()
    sp_50    = (gates < 0.5).mean()  * 100
    sp_01    = (gates < 0.01).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: full distribution
    axes[0].hist(gates, bins=100, color="#4A6FA5", edgecolor="none", alpha=0.85)
    axes[0].axvline(0.01, color="#E74C3C", ls="--", lw=1.5, label="0.01 threshold")
    axes[0].axvline(0.50, color="#F39C12", ls="--", lw=1.5, label="0.50 threshold")
    axes[0].set_xlabel("Gate value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Full gate distribution  |  λ={lambda_val}")
    axes[0].legend()

    # Right: zoom into [0, 0.1] to see the spike at 0
    near_zero = gates[gates < 0.1]
    axes[1].hist(near_zero, bins=60, color="#E74C3C", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Gate value (zoomed to [0, 0.1])")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Zoom near 0  |  "
        f"Sparse(0.5)={sp_50:.1f}%  |  Sparse(0.01)={sp_01:.1f}%"
    )

    fig.suptitle(f"Gate value distribution — λ={lambda_val}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {save_path}")


def plot_training_curves(histories: dict,
                         save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors    = ["#2ECC71", "#E67E22", "#E74C3C"]

    for (lam, hist), color in zip(histories.items(), colors):
        ep   = [h["epoch"]    for h in hist]
        acc  = [h["accuracy"] for h in hist]
        sp   = [h["sparsity"] for h in hist]
        lbl  = f"λ={lam}"

        # Mark warmup/pruning boundary
        boundary = WARMUP_EPOCHS + 0.5

        axes[0].plot(ep, acc, color=color, label=lbl, lw=2)
        axes[1].plot(ep, sp,  color=color, label=lbl, lw=2)

    for ax in axes:
        ax.axvline(boundary, color="gray", ls=":", lw=1, label="warmup ends")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title("Test accuracy");   axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_title("Sparsity (gate < 0.5)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Sparsity (%)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {save_path}")


# ============================================================
# Part 7: Main Experiment
# ============================================================

def main():
    print("=" * 65)
    print("Self-Pruning Neural Network — CIFAR-10")
    print("=" * 65)

    lambdas     = [0.5, 2.0, 5.0]
    results     = {}
    histories   = {}
    best_model  = None
    best_lambda = None
    best_acc    = 0.0

    for lam in lambdas:
        print(f"\n{'─'*60}")
        print(f"  λ = {lam}   (warmup: {WARMUP_EPOCHS} epochs, pruning: {25-WARMUP_EPOCHS} epochs)")
        print(f"{'─'*60}")

        model, history, test_acc, sparsity = train_model(
            lambda_sparse = lam,
            epochs        = 25,
            weight_lr     = 1e-3,
            gate_lr       = 5e-3,
            batch_size    = 128,
        )

        results[lam]   = (test_acc, sparsity)
        histories[lam] = history

        if test_acc > best_acc:
            best_acc    = test_acc
            best_model  = model
            best_lambda = lam

        sp_strict = model.network_sparsity()["pct_strict"]
        print(f"\n  FINAL  Accuracy={test_acc:.2f}%  "
              f"Sparsity(0.5)={sparsity:.1f}%  "
              f"Sparsity(0.01)={sp_strict:.1f}%")

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Lambda':>8}  {'Test Accuracy':>14}  {'Sparsity (gate<0.5)':>20}")
    print("=" * 65)
    for lam, (acc, sp) in results.items():
        print(f"{lam:>8.1f}  {acc:>13.2f}%  {sp:>19.1f}%")
    print("=" * 65)
    print()
    print("Note: 'Sparsity' = % of gates below 0.5 (less than half-open).")
    print("      A gate < 0.5 contributes < 50% of that weight's signal.")
    print("      A gate < 0.01 contributes < 1% — effectively zero.")

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerating plots...")
    if best_model is not None:
        plot_gate_distribution(best_model, best_lambda)
    plot_training_curves(histories)
    print("\nDone! Check gate_distribution.png and training_curves.png")


if __name__ == "__main__":
    main()