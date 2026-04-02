"""
Sensitivity analysis of the three-role model to the ambiguity radius ρ.

This script summarises the occurrence counts and key transition
probabilities across four ambiguity radius thresholds (ρ ∈ {1/8, 1/7,
1/6, 1/5}), demonstrating the robustness of the three-role
classification proposed in the article. Segments and their FQS
coordinates remain identical across thresholds; only the classification
boundary changes.

Supplementary material for:
    Pereira, S., Bernardes, G., & Oliveira Martins, J.
    "Statistically Modelling Qualia Syntax Through Fourier Space:
     A Three-role Probabilistic Framework Where Ambiguity Becomes
     Structure"
"""

import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. Raw data for each threshold
# ──────────────────────────────────────────────────────────────

QUALITIES = ["Dt", "T", "A", "O", "Wt", "C", "D"]

THRESHOLDS = {
    "1/8": {
        "counts": {"Dt": 86, "T": 70, "A": 52, "O": 33, "Wt": 31, "C": 18, "D": 12},
        "global": {
            ("Dt", "A"): 7.641196, ("Dt", "T"): 11.627907,
            ("T", "Dt"): 13.953488, ("A", "Dt"): 6.976744,
        },
        "conditional": {
            ("T", "Dt"): 60.000000, ("Dt", "A"): 26.744186,
            ("Dt", "T"): 40.697674, ("A", "Dt"): 41.176471,
            ("O", "A"): 27.272727,
        },
    },
    "1/7": {
        "counts": {"Dt": 90, "T": 67, "A": 68, "O": 31, "Wt": 30, "C": 16, "D": 11},
        "global": {
            ("Dt", "A"): 10.256410, ("Dt", "T"): 10.897436,
            ("T", "Dt"): 13.141026, ("A", "Dt"): 8.974359,
        },
        "conditional": {
            ("T", "Dt"): 61.194030, ("Dt", "A"): 35.555556,
            ("Dt", "T"): 37.777778, ("A", "Dt"): 41.791045,
            ("O", "A"): 29.032258,
        },
    },
    "1/6": {
        "counts": {"Dt": 89, "T": 62, "A": 82, "O": 30, "Wt": 28, "C": 13, "D": 9},
        "global": {
            ("Dt", "A"): 12.820513, ("Dt", "T"): 9.615385,
            ("T", "Dt"): 12.179487, ("A", "Dt"): 10.576923,
        },
        "conditional": {
            ("T", "Dt"): 61.290323, ("Dt", "A"): 44.943820,
            ("Dt", "T"): 33.707865, ("A", "Dt"): 40.740741,
            ("O", "A"): 33.333333,
        },
    },
    "1/5": {
        "counts": {"Dt": 91, "T": 56, "A": 104, "O": 25, "Wt": 28, "C": 10, "D": 7},
        "global": {
            ("Dt", "A"): 18.1250, ("Dt", "T"): 6.5625,
            ("T", "Dt"): 10.3125, ("A", "Dt"): 14.0625,
        },
        "conditional": {
            ("T", "Dt"): 58.928571, ("Dt", "A"): 63.736264,
            ("Dt", "T"): 23.076923, ("A", "Dt"): 43.689320,
            ("O", "A"): 32.000000,
        },
    },
}


# ──────────────────────────────────────────────────────────────
# 2. Build summary table
# ──────────────────────────────────────────────────────────────

def build_summary() -> pd.DataFrame:
    """Compile the key metrics across all thresholds into a single DataFrame."""

    rows = []
    rho_labels = list(THRESHOLDS.keys())

    for rho in rho_labels:
        d = THRESHOLDS[rho]
        counts = d["counts"]
        g = d["global"]
        m = d["conditional"]

        rows.append({
            "ρ": rho,
            # Occurrence counts
            "Dt": counts["Dt"],
            "T": counts["T"],
            "A": counts["A"],
            "O": counts["O"],
            "Wt": counts["Wt"],
            "C": counts["C"],
            "D": counts["D"],
            "T + A": counts["T"] + counts["A"],
            # Key global transition probabilities (%)
            "G¹(Dt→A)": g[("Dt", "A")],
            "G¹(Dt→T)": g[("Dt", "T")],
            "G¹(T→Dt)": g[("T", "Dt")],
            "G¹(A→Dt)": g[("A", "Dt")],
            # Key conditional probabilities (%)
            "M¹(T,Dt)": m[("T", "Dt")],
            "M¹(Dt,A)": m[("Dt", "A")],
            "M¹(Dt,T)": m[("Dt", "T")],
            "M¹(A,Dt)": m[("A", "Dt")],
            "M¹(O,A)": m[("O", "A")],
            # Aggregate mediating share of Dt departures (%)
            "M¹(Dt,T) + M¹(Dt,A)": m[("Dt", "T")] + m[("Dt", "A")],
        })

    return pd.DataFrame(rows).set_index("ρ")


# ──────────────────────────────────────────────────────────────
# 3. Display
# ──────────────────────────────────────────────────────────────

def print_section(title: str, df: pd.DataFrame, fmt: str = ".1f") -> None:
    """Print a titled subsection of the summary table."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    print(df.to_string(float_format=lambda x: f"{x:{fmt}}"))


def main() -> None:
    summary = build_summary()

    print("=" * 72)
    print("  SENSITIVITY ANALYSIS: AMBIGUITY RADIUS ρ")
    print("  Three-role model robustness across ρ ∈ {1/8, 1/7, 1/6, 1/5}")
    print("=" * 72)

    # Occurrence counts
    count_cols = ["Dt", "T", "A", "O", "Wt", "C", "D", "T + A"]
    print_section(
        "Occurrence counts (after repeated-quality deletion)",
        summary[count_cols],
        fmt=".0f",
    )

    # Key global transition probabilities
    global_cols = ["G¹(Dt→A)", "G¹(Dt→T)", "G¹(T→Dt)", "G¹(A→Dt)"]
    print_section(
        "Key global transition probabilities G¹ (%)",
        summary[global_cols],
        fmt=".2f",
    )

    # Key conditional probabilities
    cond_cols = ["M¹(T,Dt)", "M¹(Dt,A)", "M¹(Dt,T)", "M¹(A,Dt)", "M¹(O,A)"]
    print_section(
        "Key conditional probabilities M¹ (%)",
        summary[cond_cols],
        fmt=".1f",
    )

    # Aggregate mediating share
    agg_cols = ["M¹(Dt,T) + M¹(Dt,A)"]
    print_section(
        "Aggregate mediating share of Dt departures (%)",
        summary[agg_cols],
        fmt=".1f",
    )

    print(f"\n{'─' * 72}")
    print("  INTERPRETATION NOTES")
    print(f"{'─' * 72}")
    print(
        "  • Diatonicity counts are virtually invariant (86–91), confirming\n"
        "    that diatonic segments lie far from the ambiguity boundary.\n"
        "  • M¹(T,Dt) — the anchor role's chief signature — never falls\n"
        "    below 58.9%, remaining the highest single value in M¹.\n"
        "  • Varying ρ redistributes segments within the mediating layer\n"
        "    (T ↔ A), but their combined share of Dt departures stays high\n"
        "    (67.4%–86.8%).\n"
        "  • Colouristic qualities consistently route departures through\n"
        "    ambiguity (M¹(O,A) ranges 27.3%–33.3%), not directly to the\n"
        "    anchor.\n"
        "  • ρ = 1/5 represents the model's upper boundary: ambiguity\n"
        "    surpasses diatonicity in count (104 vs 91) and concentrates\n"
        "    63.7% of Dt's conditional departures onto a single pathway."
    )
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()