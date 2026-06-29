# Generators

Scripts that produce publication-ready figures, tables, and reports from pre-computed analysis results.

**Key Principle**: These scripts are **consumers** of pipeline outputs. They read finished results and format for presentation (manuscript, extended data, supplementary).

---

## Figure Generation

### `generate_manuscript_figures.py`

**Purpose**: Generate all figures cited in the main manuscript from Cox analysis outputs.

**Input**:
- `results/cox_prodromal_abk_{TIMESTAMP}/` — Cox model outputs
- Pre-computed KM curves, forest plots, interaction data

**Output**:
- `docs/publication/figures/` directory:
  - `Figure_1a_KM_RBD_PD.pdf` — Kaplan-Meier curves by RBD risk group
  - `Figure_1b_forest_cross_outcome.pdf` — Forest plot across all outcomes
  - `Figure_2a_interaction_heatmap.pdf` — RBD × PRS interaction heatmap
  - `Figure_2b_RERI_forest.pdf` — RERI (synergy) forest plot
  - `Figure_3a_threshold_stability.pdf` — Risk threshold stability analysis
  - `Figure_3b_CIF_vs_KM.pdf` — Cumulative incidence vs. KM curves

**Run**:
```bash
python generators/generate_manuscript_figures.py
```

**Dependencies**:
- Must run after `pipelines/run_cox_pipeline.py` completes
- Reads from timestamped results directory (specify in script header)

---

### `generate_model_forest_plots.py`

**Purpose**: Generate detailed forest plots for individual Cox models.

**Input**:
- Cox model coefficient estimates and CIs
- Hazard ratios by outcome and risk group

**Output**:
- `results/cox_prodromal_abk_{TIMESTAMP}/forest_plots/`:
  - `model_0_rbd_only.pdf` — RBD unadjusted model
  - `model_a_rbd_prs.pdf` — RBD + PRS adjustment
  - `model_f_rbd_prs_interaction.pdf` — RBD × PRS interaction
  - `model_4_competing_risks.pdf` — Competing risks analysis

**Features**:
- Log-scale x-axis
- Reference line at HR = 1.0
- Stratification by risk group and outcome
- Proper CI formatting and styling

**Run**:
```bash
python generators/generate_model_forest_plots.py
```

---

### `generate_rbd_spline_plot.py`

**Purpose**: Visualize non-linear RBD → PD relationship using spline models.

**Input**:
- Cox spline model predictions
- RBD score distribution

**Output**:
- `results/cox_prodromal_abk_{TIMESTAMP}/spline_plot.pdf`

**Content**:
- Smooth spline curve showing HR as function of continuous RBD score
- 95% CI bounds
- Rug plot of RBD distribution
- Risk group threshold annotations

**Run**:
```bash
python generators/generate_rbd_spline_plot.py
```

---

## Table Generation

### `generate_formal_tables.py`

**Purpose**: Format Cox model results as publication-ready tables.

**Input**:
- Model coefficient estimates, CIs, p-values
- Outcome-specific results
- Risk group stratifications

**Output**:
- `docs/publication/tables/`:
  - `Table_1_Cohort_Characteristics.xlsx` — Demographics by RBD group
  - `Table_2_Cox_Models.xlsx` — Main Cox model results
  - `Table_S1_Model_Diagnostics.xlsx` — PH tests, C-indices
  - `Table_S2_Risk_Group_Stratification.xlsx` — Event rates by group
  - `Table_S3_Competing_Risks.xlsx` — Competing risks model results
  - `Table_S4_Interaction_Analysis.xlsx` — Interaction terms

**Formatting**:
- Hazard ratios with 95% CIs in (X.XX, X.XX–X.XX) format
- P-values with appropriate significance notation (*, **, ***)
- Footnotes explaining methods
- Outcome-specific headers

**Run**:
```bash
python generators/generate_formal_tables.py
```

---

### `generate_consort_table.py`

**Purpose**: Generate CONSORT flow diagram showing cohort assembly.

**Input**:
- `data/pp/res_build_final_dataset/` — Dataset construction logs
- Exclusion counts at each stage

**Output**:
- `docs/publication/supplementary/CONSORT_tree.txt` — ASCII flow diagram
- `docs/publication/supplementary/CONSORT_tree.pdf` — Formatted PDF

**Flow**:
```
UK Biobank Cohort (n=500,000)
├─ With actigraphy data (n=100,000)
├─ After quality control filters (n=87,421)
├─ Prevalent PD excluded (n=449)
└─ Final incident case cohort (n=86,972 control + 448 cases)
```

**Run**:
```bash
python generators/generate_consort_table.py
```

---

## Logging & Reports

### `generate_logs.py`

**Purpose**: Create comprehensive analysis logs documenting pipeline execution.

**Input**:
- Timestamped output directories
- Pipeline configuration
- Analysis summary statistics

**Output**:
- `results/cox_prodromal_abk_{TIMESTAMP}/analysis_log.txt` — Detailed execution log
- `results/cox_prodromal_abk_{TIMESTAMP}/summary_stats.json` — JSON summary

**Content**:
- Pipeline stage start/end times
- Data quality metrics (missing rates, event counts)
- Model fit statistics (iterations, convergence)
- Number of bootstrap replications completed
- File manifest and output locations

**Run**:
```bash
python generators/generate_logs.py
```

---

## Workflow

### Typical Publication Figure/Table Workflow

1. **Run analysis pipeline**: `pipelines/run_cox_pipeline.py`
   - Generates `results/cox_prodromal_abk_{TIMESTAMP}/`

2. **Generate main figures**: `generators/generate_manuscript_figures.py`
   - Creates Figure 1, 2, 3 in `docs/publication/figures/`

3. **Generate main tables**: `generators/generate_formal_tables.py`
   - Creates Table 1, 2 in `docs/publication/tables/`

4. **Generate supplementary**: `generators/generate_rbd_spline_plot.py`
   - Creates extended data figures

5. **Document results**: `generators/generate_logs.py`
   - Creates analysis log and summary statistics

### Batch Execution

Run all generators in sequence:
```bash
python generators/generate_manuscript_figures.py && \
python generators/generate_model_forest_plots.py && \
python generators/generate_formal_tables.py && \
python generators/generate_consort_table.py && \
python generators/generate_logs.py
```

---

## Configuration

Edit generator scripts to specify:
- **Result directory**: Where to read Cox outputs from
- **Output directory**: Where to write figures/tables
- **Figure formats**: PDF/PNG/SVG
- **Figure styling**: Colors, fonts, dimensions
- **Table formatting**: Decimal places, significant digits

Example:
```python
RESULTS_DIR = Path("results/cox_prodromal_abk_2026_03_27_13_06_48")
PUB_DIR = Path("docs/publication")
FIG_DPI = 300
TABLE_DECIMALS = 2
```

---

## Quality Checklist

Before committing figures/tables to manuscript:

- [ ] **Figures**: Check axis labels, legends, reference lines
- [ ] **Forest plots**: Verify HR scale (log), point estimates, CIs aligned
- [ ] **Tables**: Verify column headers, footnotes, p-value formatting
- [ ] **Consistency**: KM curves, forest plots, and tables should agree
- [ ] **Colors**: Matches publication color scheme
- [ ] **Resolution**: 300 DPI for publication-quality figures

---

## Troubleshooting

**"Results directory not found"**: Update `RESULTS_DIR` in script to match timestamped output from cox pipeline.

**Missing data in figures**: Ensure Cox pipeline completed all models. Check `cox_config.py` to verify models are enabled.

**Table formatting issues**: Verify outcome names and risk group labels match between cox analysis and generator script.

**PDF generation fails**: Check matplotlib backend is set to "Agg" (non-interactive).
