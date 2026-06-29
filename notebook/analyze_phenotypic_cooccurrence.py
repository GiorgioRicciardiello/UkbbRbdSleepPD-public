"""
Phenotypic Co-occurrence Analysis: RBD-Stratified Prodromal Marker Clustering

Identifies prodromal marker combination patterns in high-RBD controls.
Answers: What % have isolated vs clustered markers? Is there an autonomic syndrome?

Author: Research Analysis Pipeline
Date: 2026-06-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(
    r"C:\Users\riccig01\OneDrive\Projects\MtSinai\During\UkbbRbdSleepPD"
)

DATA_PATH = PROJECT_DIR / "data" / "pp" / "res_build_final_dataset" / "ehr_diag_pd_rbd_only_all.parquet"

OUTPUT_DIR = PROJECT_DIR / "notebook" / "results" / "phenotypic_cooccurrence_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODROMAL_MARKERS = [
    "constipation",
    "depression",
    "anxiety",
    "orthostatic",
    "erectile_dysfunction",
    "dream_enactment",
    "anosmia",
    "hyposmia",
]

# Marker columns use "prodromal_{marker}_post" naming
MARKER_COLS_MAPPING = {m: f"prodromal_{m}_post" for m in PRODROMAL_MARKERS}

# Domain groupings
DOMAIN_MAP = {
    "autonomic": ["constipation", "orthostatic", "erectile_dysfunction"],
    "mood": ["depression", "anxiety"],
    "olfactory": ["anosmia", "hyposmia"],
    "dream": ["dream_enactment"],
}

# ============================================================================
# 1. LOAD & FILTER DATA
# ============================================================================


def load_controls() -> pd.DataFrame:
    """Load control subjects with valid actigraphy, baseline & follow-up prodromality."""
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    # Deduplicate by eid (keep first)
    df = df.groupby("eid").first().reset_index()
    print(f"  After deduplication: {len(df):,}")

    # Filter controls
    df = df[df["control"] == True].copy()
    print(f"  Controls: {len(df):,}")

    # Exclude training set & neuro exclusions
    df = df[(df["train_sleep"] != True) & (df["neuro_exclude"] == 0)].copy()
    print(f"  After exclusions: {len(df):,}")

    return df


# ============================================================================
# 2. SINGLE MARKER PREVALENCE
# ============================================================================


def compute_single_marker_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prevalence of each marker by RBD group (post-baseline)."""
    print("\n" + "=" * 70)
    print("SINGLE MARKER PREVALENCE (Follow-up)")
    print("=" * 70)

    results = []

    for marker, col_post in MARKER_COLS_MAPPING.items():
        if col_post not in df.columns:
            print(f"  Warning: {col_post} not found")
            continue

        # Overall
        n_total = df[col_post].notna().sum()
        n_yes = (df[col_post] == 1).sum()
        pct_overall = 100 * n_yes / n_total if n_total > 0 else 0

        # By RBD group
        for rbd_group in ["Low", "Mid", "High"]:
            subset = df[df["rg_pctl3"] == rbd_group]
            n = (subset[col_post] == 1).sum()
            denom = subset[col_post].notna().sum()
            pct = 100 * n / denom if denom > 0 else 0

            results.append(
                {
                    "marker": marker,
                    "rbd_group": rbd_group,
                    "n_affected": int(n),
                    "n_total": int(denom),
                    "pct_affected": round(pct, 2),
                }
            )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "single_marker_prevalence.csv", index=False)
    print(result_df.to_string())

    return result_df


# ============================================================================
# 3. PAIRWISE CO-OCCURRENCE
# ============================================================================


def compute_pairwise_cooccurrence(df: pd.DataFrame) -> dict:
    """Compute crosstabs for all marker pairs by RBD group."""
    print("\n" + "=" * 70)
    print("PAIRWISE CO-OCCURRENCE")
    print("=" * 70)

    marker_pairs = list(combinations(PRODROMAL_MARKERS, 2))
    pairwise_results = {}

    for marker1, marker2 in marker_pairs:
        col1 = MARKER_COLS_MAPPING[marker1]
        col2 = MARKER_COLS_MAPPING[marker2]

        if col1 not in df.columns or col2 not in df.columns:
            continue

        print(f"\n{marker1.upper()} × {marker2.upper()}")
        pair_key = f"{marker1}_x_{marker2}"

        # For each RBD group
        pair_data = {}
        for rbd_group in ["Low", "Mid", "High"]:
            subset = df[df["rg_pctl3"] == rbd_group]

            # Create contingency table
            crosstab = pd.crosstab(subset[col1], subset[col2], margins=False)
            print(f"\n  {rbd_group}:")
            print(crosstab)

            # Co-occurrence: both markers present
            both = ((subset[col1] == 1) & (subset[col2] == 1)).sum()
            only_1 = ((subset[col1] == 1) & (subset[col2] != 1)).sum()
            only_2 = ((subset[col1] != 1) & (subset[col2] == 1)).sum()
            neither = ((subset[col1] != 1) & (subset[col2] != 1)).sum()

            denom = subset[col1].notna().sum()

            pair_data[rbd_group] = {
                "both": both,
                "only_1": only_1,
                "only_2": only_2,
                "neither": neither,
                "total": denom,
                "pct_both": 100 * both / denom if denom > 0 else 0,
            }

        pairwise_results[pair_key] = pair_data

        # Save individual crosstabs
        crosstab_df = pd.DataFrame(pair_data).T
        crosstab_df.to_csv(OUTPUT_DIR / f"crosstab_{pair_key}.csv")

    return pairwise_results


# ============================================================================
# 4. MULTI-MARKER PHENOTYPES
# ============================================================================


def compute_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    """Define and count phenotypic patterns."""
    print("\n" + "=" * 70)
    print("MULTI-MARKER PHENOTYPIC PATTERNS")
    print("=" * 70)

    # Build marker presence matrix
    marker_cols = [MARKER_COLS_MAPPING[m] for m in PRODROMAL_MARKERS if MARKER_COLS_MAPPING[m] in df.columns]

    # Isolate pattern: count markers
    df["n_markers"] = df[marker_cols].sum(axis=1).astype(int)

    # Domain dominance: which domain(s) have markers?
    def get_domain_pattern(row: pd.Series) -> str:
        domains_present = []
        for domain, markers in DOMAIN_MAP.items():
            cols = [MARKER_COLS_MAPPING[m] for m in markers if m in MARKER_COLS_MAPPING and MARKER_COLS_MAPPING[m] in df.columns]
            if any(row[c] == 1 for c in cols if c in row.index):
                domains_present.append(domain)

        if not domains_present:
            return "silent"
        elif len(domains_present) == 1:
            return f"{domains_present[0]}_only"
        else:
            return "mixed_" + "_".join(sorted(domains_present))

    df["phenotype"] = df.apply(get_domain_pattern, axis=1)

    # Count phenotypes by RBD group
    phenotype_counts = []

    for rbd_group in ["Low", "Mid", "High"]:
        subset = df[df["rg_pctl3"] == rbd_group]
        n_total = len(subset)

        # Isolation pattern
        for n_m in range(0, subset["n_markers"].max() + 1):
            count = (subset["n_markers"] == n_m).sum()
            pct = 100 * count / n_total
            phenotype_counts.append(
                {
                    "rbd_group": rbd_group,
                    "phenotype_type": "isolation",
                    "phenotype_name": f"{n_m}_markers",
                    "count": count,
                    "n_group": n_total,
                    "pct": round(pct, 2),
                }
            )

        # Domain pattern
        for phenotype in subset["phenotype"].unique():
            count = (subset["phenotype"] == phenotype).sum()
            pct = 100 * count / n_total
            phenotype_counts.append(
                {
                    "rbd_group": rbd_group,
                    "phenotype_type": "domain_dominance",
                    "phenotype_name": phenotype,
                    "count": count,
                    "n_group": n_total,
                    "pct": round(pct, 2),
                }
            )

    phenotype_df = pd.DataFrame(phenotype_counts)

    # Separate outputs
    isolation_df = phenotype_df[phenotype_df["phenotype_type"] == "isolation"]
    domain_df = phenotype_df[phenotype_df["phenotype_type"] == "domain_dominance"]

    print("\nISOLATION PATTERN (by marker count):")
    print(isolation_df.to_string(index=False))

    print("\nDOMAIN DOMINANCE PATTERN:")
    print(domain_df.to_string(index=False))

    isolation_df.to_csv(OUTPUT_DIR / "phenotype_isolation_pattern.csv", index=False)
    domain_df.to_csv(OUTPUT_DIR / "phenotype_domain_dominance.csv", index=False)
    phenotype_df.to_csv(OUTPUT_DIR / "phenotype_all_patterns.csv", index=False)

    return df, phenotype_df


# ============================================================================
# 5. KEY CLUSTERS: AUTONOMIC SYNDROME
# ============================================================================


def analyze_autonomic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Focus on autonomic syndrome: constipation + orthostatic combinations."""
    print("\n" + "=" * 70)
    print("AUTONOMIC SYNDROME CLUSTERING")
    print("=" * 70)

    results = []
    col_const = MARKER_COLS_MAPPING["constipation"]
    col_ortho = MARKER_COLS_MAPPING["orthostatic"]

    for rbd_group in ["Low", "Mid", "High"]:
        subset = df[df["rg_pctl3"] == rbd_group]
        n_total = len(subset)

        const = subset[col_const] == 1
        ortho = subset[col_ortho] == 1

        # Four patterns
        both = (const & ortho).sum()
        only_const = (const & ~ortho).sum()
        only_ortho = (~const & ortho).sum()
        neither = (~const & ~ortho).sum()

        results.append(
            {
                "rbd_group": rbd_group,
                "pattern": "both_constipation_orthostatic",
                "count": both,
                "pct": round(100 * both / n_total, 2),
                "n_total": n_total,
            }
        )
        results.append(
            {
                "rbd_group": rbd_group,
                "pattern": "only_constipation",
                "count": only_const,
                "pct": round(100 * only_const / n_total, 2),
                "n_total": n_total,
            }
        )
        results.append(
            {
                "rbd_group": rbd_group,
                "pattern": "only_orthostatic",
                "count": only_ortho,
                "pct": round(100 * only_ortho / n_total, 2),
                "n_total": n_total,
            }
        )
        results.append(
            {
                "rbd_group": rbd_group,
                "pattern": "neither",
                "count": neither,
                "pct": round(100 * neither / n_total, 2),
                "n_total": n_total,
            }
        )

    autonomic_df = pd.DataFrame(results)
    print(autonomic_df.to_string(index=False))
    autonomic_df.to_csv(OUTPUT_DIR / "autonomic_syndrome_clustering.csv", index=False)

    return autonomic_df


# ============================================================================
# 6. VISUALIZATION: HEATMAP & BAR CHARTS
# ============================================================================


def plot_marker_cooccurrence_heatmap(df: pd.DataFrame) -> None:
    """Generate co-occurrence heatmap (High-RBD vs Low-RBD)."""
    print("\n" + "=" * 70)
    print("GENERATING HEATMAP")
    print("=" * 70)

    marker_cols = [MARKER_COLS_MAPPING[m] for m in PRODROMAL_MARKERS if MARKER_COLS_MAPPING[m] in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    for idx, rbd_group in enumerate(["Low", "High"]):
        subset = df[df["rg_pctl3"] == rbd_group][marker_cols]

        # Co-occurrence matrix
        cooc_matrix = subset.T.dot(subset)
        cooc_pct = 100 * cooc_matrix / len(subset)

        # Normalize diagonal to 100%
        np.fill_diagonal(cooc_pct.values, 100)

        # Shorten labels
        labels = [m.replace("_", " ").title() for m in PRODROMAL_MARKERS]

        ax = axes[idx]
        sns.heatmap(
            cooc_pct,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "% Co-occurrence"},
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=100,
        )
        ax.set_title(f"Prodromal Marker Co-occurrence: {rbd_group}-RBD (n={len(subset):,})")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cooccurrence_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: cooccurrence_heatmap.png")
    plt.close()


def plot_phenotype_distribution(phenotype_df: pd.DataFrame) -> None:
    """Bar charts: phenotype distribution by RBD group."""
    print("Generating phenotype distribution chart...")

    domain_df = phenotype_df[phenotype_df["phenotype_type"] == "domain_dominance"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Prepare data for grouped bar chart
    pivot_data = domain_df.pivot_table(
        index="phenotype_name", columns="rbd_group", values="pct", fill_value=0
    )

    # Reorder columns
    pivot_data = pivot_data[["Low", "Mid", "High"]]

    pivot_data.plot(kind="bar", ax=ax, width=0.8, color=["#1f77b4", "#ff7f0e", "#d62728"])

    ax.set_title("Prodromal Phenotype Distribution by RBD Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Phenotype (Domain Dominance)", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(title="RBD Group", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phenotype_distribution.png", dpi=150, bbox_inches="tight")
    print("  Saved: phenotype_distribution.png")
    plt.close()


def plot_autonomic_patterns(autonomic_df: pd.DataFrame) -> None:
    """Bar chart: autonomic syndrome patterns."""
    print("Generating autonomic patterns chart...")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    patterns = ["both_constipation_orthostatic", "only_constipation", "only_orthostatic", "neither"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    x = np.arange(len(["Low", "Mid", "High"]))
    width = 0.2

    for i, pattern in enumerate(patterns):
        subset = autonomic_df[autonomic_df["pattern"] == pattern]
        values = subset["pct"].values
        ax.bar(x + i * width, values, width, label=pattern.replace("_", " ").title(), color=colors[i])

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("RBD Group", fontsize=12)
    ax.set_title("Autonomic Syndrome Phenotypes by RBD Group", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autonomic_syndrome_patterns.png", dpi=150, bbox_inches="tight")
    print("  Saved: autonomic_syndrome_patterns.png")
    plt.close()


# ============================================================================
# 7. STATISTICAL TESTS: CHI-SQUARE FOR RBD STRATIFICATION
# ============================================================================


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether phenotype distributions differ by RBD group."""
    print("\n" + "=" * 70)
    print("CHI-SQUARE TESTS: Phenotype × RBD Group")
    print("=" * 70)

    results = []

    # Test isolation patterns
    max_markers = int(np.nanmax(df["n_markers"].values)) if df["n_markers"].notna().any() else 0
    for n_m in range(0, max_markers + 1):
        contingency = []
        for rbd_group in ["Low", "Mid", "High"]:
            subset = df[df["rg_pctl3"] == rbd_group]
            count = (subset["n_markers"] == n_m).sum()
            contingency.append(count)

        if sum(contingency) > 0:
            # Chi-square (3 groups)
            cont_table = pd.DataFrame(
                {
                    "rbd_group": ["Low", "Mid", "High"],
                    "has_pattern": contingency,
                    "no_pattern": [
                        len(df[df["rg_pctl3"] == g]) - c
                        for g, c in zip(["Low", "Mid", "High"], contingency)
                    ],
                }
            )
            chi2, p, dof, expected = chi2_contingency(
                cont_table[["has_pattern", "no_pattern"]].values
            )

            results.append(
                {
                    "phenotype": f"{n_m}_markers",
                    "phenotype_type": "isolation",
                    "chi2": round(chi2, 4),
                    "p_value": round(p, 4),
                    "dof": dof,
                }
            )

    # Test domain patterns
    for phenotype in df["phenotype"].unique():
        contingency = []
        for rbd_group in ["Low", "Mid", "High"]:
            subset = df[df["rg_pctl3"] == rbd_group]
            count = (subset["phenotype"] == phenotype).sum()
            contingency.append(count)

        if sum(contingency) > 1:  # Need at least 2 occurrences
            cont_table = pd.DataFrame(
                {
                    "rbd_group": ["Low", "Mid", "High"],
                    "has_pattern": contingency,
                    "no_pattern": [
                        len(df[df["rg_pctl3"] == g]) - c
                        for g, c in zip(["Low", "Mid", "High"], contingency)
                    ],
                }
            )
            chi2, p, dof, expected = chi2_contingency(
                cont_table[["has_pattern", "no_pattern"]].values
            )

            results.append(
                {
                    "phenotype": phenotype,
                    "phenotype_type": "domain",
                    "chi2": round(chi2, 4),
                    "p_value": round(p, 4),
                    "dof": dof,
                }
            )

    chi_df = pd.DataFrame(results)
    chi_df = chi_df.sort_values("p_value")
    print(chi_df.to_string(index=False))
    chi_df.to_csv(OUTPUT_DIR / "chi_square_phenotype_rbd.csv", index=False)

    return chi_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Execute full phenotypic co-occurrence analysis."""
    print("=" * 70)
    print("PHENOTYPIC CO-OCCURRENCE ANALYSIS")
    print("Controls with Valid Actigraphy")
    print("=" * 70)

    # Load data
    df = load_controls()
    print(f"Final cohort: {len(df):,} controls")

    # Single markers
    marker_prev = compute_single_marker_prevalence(df)

    # Pairwise
    pairwise = compute_pairwise_cooccurrence(df)

    # Phenotypes
    df_pheno, phenotype_results = compute_phenotypes(df)

    # Autonomic syndrome focus
    autonomic_results = analyze_autonomic_clusters(df)

    # Chi-square tests
    chi_results = chi_square_tests(df_pheno)

    # Visualizations
    plot_marker_cooccurrence_heatmap(df)
    plot_phenotype_distribution(phenotype_results)
    plot_autonomic_patterns(autonomic_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
