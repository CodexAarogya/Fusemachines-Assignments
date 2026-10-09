"""Nightly regression check for the PRODUCTION config (prompt_v3, min_sources=2).
Exit 0 = healthy, exit 2 = pass-rate dropped more than MAX_DROP points below the baseline -> alert."""
import json, sys
from pathlib import Path
from mlops.run_experiments import run_version, ROOT
BASELINE_FILE, MAX_DROP = ROOT / "reports" / "baseline.json", 10.0

def main():
    _, df, _ = run_version("v3", 2)
    pct = 100 * ((df.correctness == "correct").sum() + (df.disclosure == "correct").sum()) / (2 * len(df))
    if not BASELINE_FILE.exists():
        BASELINE_FILE.write_text(json.dumps({"pct_tests_passed": pct})); print(f"baseline set: {pct:.2f}"); return
    base = json.loads(BASELINE_FILE.read_text())["pct_tests_passed"]
    print(f"pct_tests_passed={pct:.2f} baseline={base:.2f}")
    if base - pct > MAX_DROP:
        print("ALERT: regression beyond threshold"); sys.exit(2)

if __name__ == "__main__":
    main()
