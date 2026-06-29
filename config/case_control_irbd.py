"""
Design: Nested case–control within a sleep-clinic cohort.

Cases: iRBD.

Controls: Non-iRBD clinic patients (exclude mimics, unclear if desired).

Rarity: iRBD is rare → matching improves efficiency.

Referral structure: All subjects are PSG-referred → reduces referral bias.

Matching goal: Balance age and sex.

Sampling: Without replacement (no duplicated controls).

Analysis implication: Requires conditional logistic regression (or stratified methods) downstream, because matching induces selection bias if ignore

Exact matching strategy (epidemiologically correct)
Matching variables
- Sex: exact
- Age: ±2 years (standard)

Ratio
- 1:1 or 1:2 recommended for small samples
- Gains beyond 1:4 are minimal and unstable for rare exposures

Introduction to Matching in Cas…
- No replacement
- Each control used once only

"""
import pandas as pd
import pandas as pd
import numpy as np
import re

def parse_age_sex(val: str) -> tuple[int | None, str | None]:
    if pd.isna(val):
        return None, None

    s = val.strip().lower()

    # sex
    if "female" in s or re.search(r"\bf\b", s):
        sex = "F"
    elif "male" in s or re.search(r"\bm\b", s):
        sex = "M"
    else:
        sex = None

    # age
    age_match = re.search(r"\b(\d{2})\b", s)
    age = int(age_match.group(1)) if age_match else None

    return age, sex

def classify_group(dx: str) -> str | None:
    dx = dx.strip().lower()

    if "irbd" in dx :
        return "case"
    # if dx in {
    #     "prbd",
    #     "narcolepsy-prbd",
    #     "pd-rbd",
    #     "dlb-rbd",
    #     "mimic",
    #     "trauma-associated sleep disorder",
    #     "unclear",
    # }:
    #     return "control"

    if 'control' in dx:
        return 'control'

    return None

def match_irbd_controls(
    df: pd.DataFrame,
    age_tol: int = 2,
    ratio: int = 1,
) -> pd.DataFrame:
    cases = df[df["group"] == "case"].copy()
    controls = df[df["group"] == "control"].copy()

    matched_rows = []
    used_controls = set()

    for _, case in cases.iterrows():
        eligible = controls[
            (controls["sex"] == case["sex"]) &
            (controls["age"].between(case["age"] - age_tol,
                                      case["age"] + age_tol)) &
            (~controls.index.isin(used_controls))
        ]

        if eligible.empty:
            continue

        selected = eligible.sample(
            n=min(ratio, len(eligible)),
            random_state=42
        )

        used_controls.update(selected.index)

        matched_rows.append(case.to_frame().T)
        matched_rows.append(selected)

    matched_df = pd.concat(matched_rows).reset_index(drop=True)

    return matched_df


if __name__ == "__main__":
    df = pd.read_excel(r"C:\Users\riccig01\Downloads\RBD Research Tracker.xlsx",
                       sheet_name="Current Patients (cleaned up 2)")

    df[["age", "sex"]] = (
        df["Age, Sex"]
        .apply(parse_age_sex)
        .apply(pd.Series)
    )

    # pre-process diagnosis columns
    df["Diagnosis"] = df["Diagnosis"].str.strip().str.lower().fillna("")

    # make a case-control group, only 2 labels
    df["group"] = df["Diagnosis"].apply(classify_group)
    df = df.dropna(subset=["group"])
    # match irbd cases to controls
    df_matched = match_irbd_controls(df,age_tol=5, ratio=1)
    print(df_matched)

    from tabulate import tabulate

    table = tabulate(
        df["Diagnosis"].value_counts().reset_index(),
        headers=["Diagnosis", "Count"],
        tablefmt="psql",
        showindex=False,
    )

    print(table)
    df_tab = pd.DataFrame( df["Diagnosis"].value_counts().reset_index())
    df_tab.to_excel(r"C:\Users\riccig01\Downloads\diagnosis_counts.xlsx", index=True)


















