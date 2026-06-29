"""
Direct IDE entry point for Cox prodromal analysis pipeline.

Run from IDE: F5 (PyCharm) or Code Runner, without terminal.

Output: results/cox_prodromal_abk/
         results/table_one/
"""
if __name__ == "__main__":
    from library.cox_prodromal.runner import run_prodromal_pipeline

    print("\n" + "="*72)
    print("  STARTING COX PRODROMAL PIPELINE")
    print("="*72 + "\n")

    run_prodromal_pipeline()

    print("\n" + "="*72)
    print("  PIPELINE COMPLETE")
    print("="*72)