import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PAIR_NAMES = [
    "(0,1)",
    "(0,2)",
    "(0,3)",
    "(1,2)",
    "(1,3)",
    "(2,3)",
]


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def load_cache(path):
    """
    PyTorch 버전에 따라 weights_only 기본값 차이를 피하기 위해
    weights_only=False를 명시적으로 시도한다.
    """
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")

    return cache


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def flatten_1d(x, name):
    x = to_numpy(x)
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(
            f"{name} must be 1D after squeeze, "
            f"but got shape={x.shape}"
        )

    return x


def percentile_dict(x):
    x = np.asarray(x)

    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    values = np.percentile(x, percentiles)

    return {
        p: v for p, v in zip(percentiles, values)
    }


def print_percentiles(title, x):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    values = percentile_dict(x)

    for p, value in values.items():
        print(f"{p:>3}% : {value:.8f}")


def bootstrap_mean_difference(
    a,
    b,
    n_boot=5000,
    seed=42,
):
    """
    b_mean - a_mean 의 bootstrap 95% CI
    """
    a = np.asarray(a)
    b = np.asarray(b)

    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)

        diffs[i] = sample_b.mean() - sample_a.mean()

    lower, upper = np.percentile(diffs, [2.5, 97.5])

    return {
        "difference": b.mean() - a.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def group_summary(
    name,
    mask,
    current_losses,
    oracle_losses,
    oracle_regret,
    best_second_margin,
    worst_best_gap,
    oracle_correct,
):
    n = int(mask.sum())

    if n == 0:
        print(f"\n{name}: no samples")
        return None

    regret = oracle_regret[mask]
    margin = best_second_margin[mask]
    worst_gap = worst_best_gap[mask]

    result = {
        "group": name,
        "n": n,

        "current_ce": current_losses[mask].mean(),
        "oracle_ce": oracle_losses[mask].mean(),

        "oracle_regret_mean": regret.mean(),
        "oracle_regret_median": np.median(regret),
        "oracle_regret_q90": np.percentile(regret, 90),
        "oracle_regret_q95": np.percentile(regret, 95),
        "oracle_regret_q99": np.percentile(regret, 99),

        "best_second_margin_mean": margin.mean(),
        "best_second_margin_median": np.median(margin),
        "best_second_margin_q90": np.percentile(margin, 90),
        "best_second_margin_q95": np.percentile(margin, 95),

        "worst_best_gap_mean": worst_gap.mean(),
        "worst_best_gap_median": np.median(worst_gap),

        "oracle_accuracy": oracle_correct[mask].mean() * 100.0,
    }

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"N                         : {result['n']}")
    print(f"Current CE                : {result['current_ce']:.8f}")
    print(f"Oracle CE                 : {result['oracle_ce']:.8f}")
    print(f"Oracle accuracy           : {result['oracle_accuracy']:.2f}%")
    print()

    print("Oracle regret")
    print(f"  mean                    : {result['oracle_regret_mean']:.8f}")
    print(f"  median                  : {result['oracle_regret_median']:.8f}")
    print(f"  q90                     : {result['oracle_regret_q90']:.8f}")
    print(f"  q95                     : {result['oracle_regret_q95']:.8f}")
    print(f"  q99                     : {result['oracle_regret_q99']:.8f}")
    print()

    print("Best vs second-best margin")
    print(f"  mean                    : {result['best_second_margin_mean']:.8f}")
    print(f"  median                  : {result['best_second_margin_median']:.8f}")
    print(f"  q90                     : {result['best_second_margin_q90']:.8f}")
    print(f"  q95                     : {result['best_second_margin_q95']:.8f}")
    print()

    print("Worst - best gap")
    print(f"  mean                    : {result['worst_best_gap_mean']:.8f}")
    print(f"  median                  : {result['worst_best_gap_median']:.8f}")

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):

    cache_path = Path(args.cache)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found:\n{cache_path}"
        )

    print("=" * 70)
    print("Loading cache")
    print("=" * 70)
    print(cache_path)

    cache = load_cache(cache_path)

    print()
    print("Available keys:")
    for key in cache.keys():
        value = cache[key]

        if torch.is_tensor(value):
            print(
                f"  {key:<25} "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        else:
            print(
                f"  {key:<25} "
                f"type={type(value).__name__}"
            )

    required_keys = [
        "pair_losses",
        "pair_correct",
        "current_losses",
        "current_correct",
    ]

    missing = [
        k for k in required_keys
        if k not in cache
    ]

    if missing:
        raise KeyError(
            f"Missing required keys: {missing}"
        )

    # -----------------------------------------------------
    # Load tensors
    # -----------------------------------------------------

    pair_losses = to_numpy(cache["pair_losses"]).astype(np.float64)
    pair_correct = to_numpy(cache["pair_correct"]).astype(bool)

    current_losses = flatten_1d(
        cache["current_losses"],
        "current_losses",
    ).astype(np.float64)

    current_correct = flatten_1d(
        cache["current_correct"],
        "current_correct",
    ).astype(bool)

    # Expected:
    # pair_losses  : [N, 6]
    # pair_correct : [N, 6]

    if pair_losses.ndim != 2:
        raise ValueError(
            f"pair_losses must be 2D, "
            f"got {pair_losses.shape}"
        )

    if pair_losses.shape[1] != 6:
        raise ValueError(
            f"Expected 6 Mini pairs, "
            f"got {pair_losses.shape[1]}"
        )

    if pair_correct.shape != pair_losses.shape:
        raise ValueError(
            f"pair_correct shape mismatch:\n"
            f"pair_losses={pair_losses.shape}\n"
            f"pair_correct={pair_correct.shape}"
        )

    n = pair_losses.shape[0]

    if len(current_losses) != n:
        raise ValueError(
            "current_losses sample count mismatch"
        )

    if len(current_correct) != n:
        raise ValueError(
            "current_correct sample count mismatch"
        )

    if not np.isfinite(pair_losses).all():
        raise ValueError(
            "pair_losses contains NaN or Inf"
        )

    if not np.isfinite(current_losses).all():
        raise ValueError(
            "current_losses contains NaN or Inf"
        )

    # -----------------------------------------------------
    # Oracle calculation
    # -----------------------------------------------------

    oracle_pair_idx = np.argmin(
        pair_losses,
        axis=1,
    )

    sample_idx = np.arange(n)

    oracle_losses = pair_losses[
        sample_idx,
        oracle_pair_idx,
    ]

    oracle_correct = pair_correct[
        sample_idx,
        oracle_pair_idx,
    ]

    # current loss - best possible pair loss
    oracle_regret = (
        current_losses
        - oracle_losses
    )

    # -----------------------------------------------------
    # Pair loss ranking
    # -----------------------------------------------------

    sorted_losses = np.sort(
        pair_losses,
        axis=1,
    )

    best_losses = sorted_losses[:, 0]
    second_losses = sorted_losses[:, 1]
    worst_losses = sorted_losses[:, -1]

    best_second_margin = (
        second_losses
        - best_losses
    )

    worst_best_gap = (
        worst_losses
        - best_losses
    )

    # -----------------------------------------------------
    # Basic sanity
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("BASIC SANITY CHECK")
    print("=" * 70)

    print(f"N                         : {n}")
    print(f"Current CE                : {current_losses.mean():.8f}")
    print(f"Current accuracy          : {current_correct.mean() * 100:.2f}%")
    print()

    print(f"Oracle CE                 : {oracle_losses.mean():.8f}")
    print(f"Oracle accuracy           : {oracle_correct.mean() * 100:.2f}%")
    print()

    print(f"Mean Oracle regret        : {oracle_regret.mean():.8f}")
    print(f"Median Oracle regret      : {np.median(oracle_regret):.8f}")

    negative_regret = np.sum(
        oracle_regret < -1e-6
    )

    print(
        f"Regret < -1e-6 samples   : "
        f"{negative_regret}"
    )

    if negative_regret > 0:
        print()
        print(
            "[WARNING] Oracle regret should normally be >= 0."
        )
        print(
            "Check whether current_losses and pair_losses "
            "were generated from exactly the same forward setup."
        )

    # -----------------------------------------------------
    # Best static pair
    # -----------------------------------------------------

    pair_mean_losses = pair_losses.mean(axis=0)
    pair_accuracies = pair_correct.mean(axis=0) * 100.0

    print()
    print("=" * 70)
    print("STATIC PAIR RESULTS")
    print("=" * 70)

    for i, pair_name in enumerate(PAIR_NAMES):
        print(
            f"{pair_name:<6} "
            f"CE={pair_mean_losses[i]:.8f}  "
            f"Acc={pair_accuracies[i]:.2f}%"
        )

    best_static_idx = np.argmin(
        pair_mean_losses
    )

    print()
    print(
        f"Best static pair          : "
        f"{PAIR_NAMES[best_static_idx]}"
    )
    print(
        f"Best static CE            : "
        f"{pair_mean_losses[best_static_idx]:.8f}"
    )
    print(
        f"Best static accuracy      : "
        f"{pair_accuracies[best_static_idx]:.2f}%"
    )

    # =====================================================
    # EXPERIMENT A
    # Oracle regret distribution
    # =====================================================

    print_percentiles(
        "EXPERIMENT A — ORACLE REGRET PERCENTILES",
        oracle_regret,
    )

    # Positive regret concentration
    positive_regret = np.clip(
        oracle_regret,
        a_min=0.0,
        a_max=None,
    )

    total_regret = positive_regret.sum()

    print()
    print("=" * 70)
    print("ORACLE GAIN CONCENTRATION")
    print("=" * 70)

    if total_regret > 0:

        sorted_regret_desc = np.sort(
            positive_regret
        )[::-1]

        for pct in [
            1,
            5,
            10,
            20,
            25,
            50,
        ]:
            k = max(
                1,
                int(np.ceil(n * pct / 100.0)),
            )

            captured = (
                sorted_regret_desc[:k].sum()
                / total_regret
            )

            print(
                f"Top {pct:>2}% samples explain "
                f"{captured * 100:6.2f}% "
                f"of total positive regret"
            )

    # How many samples have meaningful regret?
    print()
    print("Samples above regret threshold")

    regret_thresholds = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
    ]

    for threshold in regret_thresholds:
        ratio = np.mean(
            oracle_regret > threshold
        )

        print(
            f"Regret > {threshold:<6.3f}: "
            f"{ratio * 100:6.2f}%"
        )

    # =====================================================
    # EXPERIMENT B
    # Current Correct vs Wrong
    # =====================================================

    correct_mask = current_correct
    wrong_mask = ~current_correct

    correct_summary = group_summary(
        "EXPERIMENT B — CURRENT CORRECT",
        correct_mask,
        current_losses,
        oracle_losses,
        oracle_regret,
        best_second_margin,
        worst_best_gap,
        oracle_correct,
    )

    wrong_summary = group_summary(
        "EXPERIMENT B — CURRENT WRONG",
        wrong_mask,
        current_losses,
        oracle_losses,
        oracle_regret,
        best_second_margin,
        worst_best_gap,
        oracle_correct,
    )

    # -----------------------------------------------------
    # Bootstrap correct vs wrong
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("CURRENT WRONG - CURRENT CORRECT")
    print("=" * 70)

    regret_boot = bootstrap_mean_difference(
        oracle_regret[correct_mask],
        oracle_regret[wrong_mask],
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    print("Oracle regret mean difference")
    print(
        f"Wrong - Correct             : "
        f"{regret_boot['difference']:.8f}"
    )
    print(
        f"95% bootstrap CI           : "
        f"[{regret_boot['ci_lower']:.8f}, "
        f"{regret_boot['ci_upper']:.8f}]"
    )

    margin_boot = bootstrap_mean_difference(
        best_second_margin[correct_mask],
        best_second_margin[wrong_mask],
        n_boot=args.bootstrap,
        seed=args.seed + 1,
    )

    print()
    print("Best-second margin mean difference")
    print(
        f"Wrong - Correct             : "
        f"{margin_boot['difference']:.8f}"
    )
    print(
        f"95% bootstrap CI           : "
        f"[{margin_boot['ci_lower']:.8f}, "
        f"{margin_boot['ci_upper']:.8f}]"
    )

    # =====================================================
    # EXPERIMENT C
    # Best vs second-best margin
    # =====================================================

    print_percentiles(
        "EXPERIMENT C — BEST VS SECOND-BEST MARGIN",
        best_second_margin,
    )

    print_percentiles(
        "EXPERIMENT C — WORST VS BEST GAP",
        worst_best_gap,
    )

    print()
    print("=" * 70)
    print("HOW OFTEN IS SECOND-BEST ALMOST AS GOOD?")
    print("=" * 70)

    margin_thresholds = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    for threshold in margin_thresholds:
        ratio = np.mean(
            best_second_margin <= threshold
        )

        print(
            f"margin <= {threshold:<6.3f}: "
            f"{ratio * 100:6.2f}%"
        )

    # -----------------------------------------------------
    # Oracle pair frequency
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("ORACLE PAIR FREQUENCY")
    print("=" * 70)

    for pair_idx, pair_name in enumerate(PAIR_NAMES):

        count = np.sum(
            oracle_pair_idx == pair_idx
        )

        ratio = count / n

        print(
            f"{pair_name:<6}: "
            f"{count:>4} / {n} "
            f"({ratio * 100:5.2f}%)"
        )

    # -----------------------------------------------------
    # Relation between margin and regret
    # -----------------------------------------------------

    correlation = np.corrcoef(
        oracle_regret,
        best_second_margin,
    )[0, 1]

    print()
    print("=" * 70)
    print("REGRET ↔ PAIR MARGIN")
    print("=" * 70)

    print(
        f"Pearson correlation        : "
        f"{correlation:.6f}"
    )

    # -----------------------------------------------------
    # Save per-sample results
    # -----------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame({
        "sample_idx": sample_idx,

        "current_loss": current_losses,
        "current_correct": current_correct.astype(int),

        "oracle_pair_idx": oracle_pair_idx,
        "oracle_pair": [
            PAIR_NAMES[i]
            for i in oracle_pair_idx
        ],

        "oracle_loss": oracle_losses,
        "oracle_correct": oracle_correct.astype(int),

        "oracle_regret": oracle_regret,

        "best_loss": best_losses,
        "second_best_loss": second_losses,
        "worst_loss": worst_losses,

        "best_second_margin": best_second_margin,
        "worst_best_gap": worst_best_gap,
    })

    for pair_idx, pair_name in enumerate(PAIR_NAMES):

        safe_name = (
            pair_name
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )

        df[
            f"pair_{safe_name}_loss"
        ] = pair_losses[:, pair_idx]

        df[
            f"pair_{safe_name}_correct"
        ] = pair_correct[:, pair_idx].astype(int)

    per_sample_path = (
        output_dir
        / "block1_oracle_per_sample.csv"
    )

    df.to_csv(
        per_sample_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save group summary
    # -----------------------------------------------------

    summaries = []

    if correct_summary is not None:
        summaries.append(correct_summary)

    if wrong_summary is not None:
        summaries.append(wrong_summary)

    if summaries:
        summary_df = pd.DataFrame(
            summaries
        )

        summary_path = (
            output_dir
            / "block1_correct_wrong_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

    # -----------------------------------------------------
    # Find hardest / highest regret samples
    # -----------------------------------------------------

    top_regret_df = (
        df
        .sort_values(
            "oracle_regret",
            ascending=False,
        )
        .head(args.top_samples)
    )

    top_path = (
        output_dir
        / "block1_top_oracle_regret_samples.csv"
    )

    top_regret_df.to_csv(
        top_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP ORACLE-REGRET SAMPLES")
    print("=" * 70)

    columns_to_show = [
        "sample_idx",
        "current_correct",
        "current_loss",
        "oracle_pair",
        "oracle_loss",
        "oracle_regret",
        "best_second_margin",
    ]

    print(
        top_regret_df[
            columns_to_show
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Final files
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(per_sample_path)

    if summaries:
        print(summary_path)

    print(top_path)

    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)

    print(
        """
1. Regret median이 작고,
   Top 5~10%가 total regret 대부분을 설명한다면
   → Oracle gain은 소수 difficult samples에 집중.

2. Current-wrong 그룹의 regret이
   Current-correct보다 확실히 크다면
   → pair routing은 어려운 sample에서 특히 중요.

3. best-second margin이 대부분 매우 작다면
   → exact Oracle pair accuracy는 지나치게 엄격한 metric.
   → low-regret / top-k pair prediction이 더 적절.

4. best-second margin도 크고 regret도 넓게 존재한다면
   → 좋은 pair 자체를 정확히 예측하는 selector 연구 가치가 큼.

5. regret은 큰데 best-second margin은 작다면
   → '정확한 1개 pair'보다
     나쁜 pair를 피하는 ranking 문제일 가능성이 큼.
        """
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/"
            "checkpoints/"
            "budgeted_v2_fair/"
            "pair_predictor_block1/"
            "block1_teacher_val.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/content/drive/MyDrive/"
            "mini-to-main-attention/"
            "checkpoints/"
            "budgeted_v2_fair/"
            "pair_predictor_block1/"
            "oracle_analysis"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-samples",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    main(args)