from main import run_rotating_partition_cv
if __name__ == "__main__":
    all_results, summary = run_rotating_partition_cv()
    print("\n===== SUMMARY =====")
    for k in ["TSS","HSS","accuracy","precision","recall","f1","auc"]:
        print("Mean {:9s}: {:.4f} +/- {:.4f}".format(k, summary["mean_"+k], summary["std_"+k]))
    for r in all_results:
        print("  {}: TSS={:.4f} HSS={:.4f} Acc={:.4f} F1={:.4f}".format(
            r["partition"], r["TSS"], r["HSS"], r["accuracy"], r["f1"]))
