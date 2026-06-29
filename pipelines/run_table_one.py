"""
Direct IDE entry point for Table 1 baseline characteristics.

Run from IDE: F5 (PyCharm) or Code Runner, without terminal.

Output: results/table_one/table1_rbd_risk_groups.xlsx
         results/table_one/table1_percentile_*.xlsx
         results/table_one/table1_percentile_*.csv
"""
if __name__ == "__main__":
    from library.table_one import main

    print("\n" + "="*72)
    print("  GENERATING TABLE 1 - BASELINE CHARACTERISTICS")
    print("="*72 + "\n")

    main()

    print("\n" + "="*72)
    print("  TABLE 1 COMPLETE")
    print("="*72)
