"""
List all ML pipeline runs with their timestamps and output locations.

Usage:
    python list_runs.py
"""
from pathlib import Path
from library.ml_cross_sectional.pipeline import list_all_runs

RESULTS_ROOT = Path("results/ml_cross_sectional")

if __name__ == "__main__":
    runs = list_all_runs(RESULTS_ROOT)
    
    if not runs:
        print("\nNo timestamped runs found.")
        print("\nNote: Old runs use '_final_report/' (without timestamp).")
        legacy = RESULTS_ROOT / "_final_report"
        if legacy.exists():
            print(f"  Legacy: {legacy}/")
        exit(0)
    
    print("\n" + "="*80)
    print("ML CROSS-SECTIONAL PIPELINE RUNS")
    print("="*80)
    print(f"\n{'Run ID':<30} {'Date / Time':<20} {'Output Directory'}")
    print("-"*80)
    
    for run_id, run_path in runs.items():
        # Parse run ID to extract date/time
        parts = run_id.split("_")
        date_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"
        time_str = f"{parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}"
        dt_str = f"{date_str} {time_str}"
        
        # Check what files exist
        has_fig = (run_path / "figure_feature_set_comparison.png").exists()
        has_supp = (run_path / "figure_feature_set_supplemental.png").exists()
        has_table = (run_path / "table_best_model_summary.csv").exists()
        status = "[COMPLETE]" if (has_fig and has_supp and has_table) else "[PARTIAL]"
        
        print(f"{run_id:<30} {dt_str:<20} {run_path.name} {status}")
        
        # Show files
        if has_fig:
            fig_size = (run_path / "figure_feature_set_comparison.png").stat().st_size / (1024*1024)
            print(f"  ├─ figure_feature_set_comparison.png ({fig_size:.1f} MB)")
        if has_supp:
            supp_size = (run_path / "figure_feature_set_supplemental.png").stat().st_size / (1024*1024)
            print(f"  ├─ figure_feature_set_supplemental.png ({supp_size:.1f} MB)")
        if has_table:
            print(f"  └─ table_best_model_summary.csv")
    
    print("\n" + "="*80)
    print(f"Total runs: {len(runs)}")
    print("="*80)
    print("\nNote: Each run has a unique ID (YYYYMMDD_HHMMSS_XXXXX) to avoid overwrites.")
    print("      The 5-character suffix allows multiple runs per day.")


def find_run_by_date(date_str: str, results_root: Path = RESULTS_ROOT) -> dict[str, Path]:
    """
    Find all runs on a specific date.
    
    Parameters
    ----------
    date_str : str
        Date string, e.g., "20260420" or "2026-04-20"
    
    Returns
    -------
    dict[str, Path]
        Runs matching that date, sorted by time (most recent first)
    """
    # Normalize date format
    date_str = date_str.replace("-", "")
    
    runs = list_all_runs(results_root)
    matching = {
        run_id: path for run_id, path in runs.items()
        if run_id.startswith(date_str)
    }
    return matching


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Usage: python list_runs.py 20260420
        date_arg = sys.argv[1]
        matching = find_run_by_date(date_arg)
        
        if matching:
            print(f"\nRuns on {date_arg}:")
            for run_id, path in matching.items():
                parts = run_id.split("_")
                time_str = f"{parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}"
                print(f"  {run_id}  ({time_str})  -> {path}/")
        else:
            print(f"No runs found on {date_arg}")
    else:
        # Show all runs (default)
        pass
