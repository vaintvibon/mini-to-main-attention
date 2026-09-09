import argparse
from pathlib import Path

import numpy as np
import torch


PAIR_NAMES = [
    "(0,1)",
    "(0,2)",
    "(0,3)",
    "(1,2)",
    "(1,3)",
    "(2,3)",
]


def load_cache(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def percent(x):
    return 100.0 * np.mean(x)


def print_distribution(values, title):
    values = np.asarray(values)

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    unique, counts = np.unique(
        values,
        return_counts=True,
    )

    total = len(values)

    for u, c in zip(unique, counts):
        print(
            f"{int(u)} pairs : "
            f"{c:>4} / {total} "
            f"({100.0 * c / total:6.2f}%)"
        )


def main(args):

    path = Path(args.cache)

    cache = load_cache(path)

    pair_losses = to_numpy(
        cache["pair_losses"]
    ).astype(np.float64)

    pair_correct = to_numpy(
        cache["pair_correct"]
    ).astype(bool)

    current_losses = to_numpy(
        cache["current_losses"]
    ).astype(np.float64)

    current_correct = to_numpy(
        cache["current_correct"]
    ).astype(bool)

    n = len(current_losses)

    sample_idx = np.arange(n)

    # --------------------------------------------------
    # Oracle
    # --------------------------------------------------

    oracle_pair_idx = np.argmin(
        pair_losses,
        axis=1,
    )

    oracle_losses = pair_losses[
        sample_idx,
        oracle_pair_idx,
    ]

    oracle_correct = pair_correct[
        sample_idx,
        oracle_pair_idx,
    ]

    oracle_regret = (
        current_losses
        - oracle_losses
    )

    sorted_losses = np.sort(
        pair_losses,
        axis=1,
    )

    best_second_margin = (
        sorted_losses[:, 1]
        - sorted_losses[:, 0]
    )

    # --------------------------------------------------
    # Masks
    # --------------------------------------------------

    wrong = ~current_correct
    correct = current_correct

    oracle_rescue = (
        wrong
        & oracle_correct
    )

    oracle_not_rescued = (
        wrong
        & ~oracle_correct
    )

    # At least one of 6 forced pairs predicts correctly
    any_pair_correct = np.any(
        pair_correct,
        axis=1,
    )

    correctable_by_any_pair = (
        wrong
        & any_pair_correct
    )

    # --------------------------------------------------
    # Basic
    # --------------------------------------------------

    print("=" * 70)
    print("BLOCK 1 ORACLE RESCUE ANALYSIS")
    print("=" * 70)

    print(f"Total samples              : {n}")
    print(
        f"Current correct            : "
        f"{correct.sum()} "
        f"({percent(correct):.2f}%)"
    )
    print(
        f"Current wrong              : "
        f"{wrong.sum()} "
        f"({percent(wrong):.2f}%)"
    )

    print()
    print(
        f"CE-Oracle rescued          : "
        f"{oracle_rescue.sum()}"
    )

    print(
        f"CE-Oracle rescue rate "
        f"among wrong                : "
        f"{100 * oracle_rescue.sum() / wrong.sum():.2f}%"
    )

    print()
    print(
        f"Correctable by ANY pair    : "
        f"{correctable_by_any_pair.sum()}"
    )

    print(
        f"Any-pair correction rate   : "
        f"{100 * correctable_by_any_pair.sum() / wrong.sum():.2f}%"
    )

    # Important sanity:
    # CE-min pair and accuracy-rescuing pair do not have
    # to be exactly equivalent.
    missed_by_ce_oracle = (
        correctable_by_any_pair
        & ~oracle_correct
    )

    print()
    print(
        f"Any pair correct, but "
        f"CE-oracle still wrong      : "
        f"{missed_by_ce_oracle.sum()}"
    )

    # --------------------------------------------------
    # How many pairs can rescue each wrong sample?
    # --------------------------------------------------

    num_correct_pairs = pair_correct.sum(axis=1)

    print_distribution(
        num_correct_pairs[wrong],
        "NUMBER OF CORRECT PAIRS — CURRENT WRONG",
    )

    print_distribution(
        num_correct_pairs[correctable_by_any_pair],
        "NUMBER OF CORRECT PAIRS — CORRECTABLE WRONG ONLY",
    )

    # --------------------------------------------------
    # Rescue samples: how unique is the good route?
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CE-ORACLE RESCUED SAMPLES")
    print("=" * 70)

    rescue_num_pairs = num_correct_pairs[
        oracle_rescue
    ]

    print(
        f"N                         : "
        f"{oracle_rescue.sum()}"
    )

    if oracle_rescue.sum() > 0:

        print(
            f"Mean # correct pairs      : "
            f"{rescue_num_pairs.mean():.4f}"
        )

        print(
            f"Median # correct pairs    : "
            f"{np.median(rescue_num_pairs):.4f}"
        )

        exactly_one = np.mean(
            rescue_num_pairs == 1
        )

        two_or_less = np.mean(
            rescue_num_pairs <= 2
        )

        four_or_more = np.mean(
            rescue_num_pairs >= 4
        )

        print()
        print(
            f"Exactly 1 rescuing pair   : "
            f"{exactly_one * 100:.2f}%"
        )

        print(
            f"<= 2 rescuing pairs       : "
            f"{two_or_less * 100:.2f}%"
        )

        print(
            f">= 4 rescuing pairs       : "
            f"{four_or_more * 100:.2f}%"
        )

    # --------------------------------------------------
    # Rescued vs non-rescued wrong
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("WRONG: RESCUED VS NOT RESCUED")
    print("=" * 70)

    for name, mask in [
        ("RESCUED", oracle_rescue),
        ("NOT_RESCUED", oracle_not_rescued),
    ]:

        print()
        print(name)

        if mask.sum() == 0:
            print("  no samples")
            continue

        print(
            f"  N                       : "
            f"{mask.sum()}"
        )

        print(
            f"  Current CE              : "
            f"{current_losses[mask].mean():.6f}"
        )

        print(
            f"  Oracle CE               : "
            f"{oracle_losses[mask].mean():.6f}"
        )

        print(
            f"  Oracle regret mean      : "
            f"{oracle_regret[mask].mean():.6f}"
        )

        print(
            f"  Oracle regret median    : "
            f"{np.median(oracle_regret[mask]):.6f}"
        )

        print(
            f"  Best-second mean        : "
            f"{best_second_margin[mask].mean():.6f}"
        )

        print(
            f"  Best-second median      : "
            f"{np.median(best_second_margin[mask]):.6f}"
        )

    # --------------------------------------------------
    # Each static pair: how many current errors can it fix?
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("STATIC PAIR RESCUE — CURRENT WRONG ONLY")
    print("=" * 70)

    for i, pair_name in enumerate(PAIR_NAMES):

        rescued = (
            wrong
            & pair_correct[:, i]
        )

        rescue_count = rescued.sum()

        print(
            f"{pair_name:<6} "
            f"rescues {rescue_count:>3} / {wrong.sum()} "
            f"({100 * rescue_count / wrong.sum():6.2f}%)"
        )

    # --------------------------------------------------
    # Pair frequency among CE-oracle rescued errors
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("ORACLE PAIR FREQUENCY — RESCUED ERRORS")
    print("=" * 70)

    rescued_indices = oracle_pair_idx[
        oracle_rescue
    ]

    for i, pair_name in enumerate(PAIR_NAMES):

        count = np.sum(
            rescued_indices == i
        )

        total = len(rescued_indices)

        ratio = (
            100 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"{pair_name:<6}: "
            f"{count:>3} "
            f"({ratio:6.2f}%)"
        )

    # --------------------------------------------------
    # How many pairs improve CE over current route?
    # --------------------------------------------------

    better_than_current = (
        pair_losses
        < current_losses[:, None] - 1e-7
    )

    num_better_pairs = better_than_current.sum(
        axis=1
    )

    print_distribution(
        num_better_pairs[wrong],
        "NUMBER OF PAIRS WITH LOWER CE THAN CURRENT — WRONG",
    )

    print_distribution(
        num_better_pairs[correct],
        "NUMBER OF PAIRS WITH LOWER CE THAN CURRENT — CORRECT",
    )

    # --------------------------------------------------
    # Final interpretation helper
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("WHAT TO LOOK FOR")
    print("=" * 70)

    print(
        """
A) Rescued sample에서
   'Exactly 1 rescuing pair' 비율이 높다
   → hard sample에서는 정확한 pair routing이 중요.

B) 대부분 rescued sample에
   3~6개의 correct pair가 있다
   → exact best-pair prediction은 불필요.
   → bad-pair avoidance / acceptable-set routing이 적합.

C) 한 static pair가 54개 rescue 대부분을 가져간다
   → dynamic routing 필요성이 약해질 수 있음.

D) rescued errors의 Oracle pair도 6개에 넓게 분산된다
   → input-dependent dynamic routing 근거가 강해짐.

E) 'Any pair correct'가 CE-Oracle rescue보다 훨씬 많다
   → 현재 CE-min Oracle 정의와
     accuracy-oriented routing 목표를 분리해서 봐야 함.
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

    args = parser.parse_args()

    main(args)