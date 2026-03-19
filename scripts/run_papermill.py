"""Script to run all notebooks sequentially using papermill."""

import os
import sys
import papermill as pm
from pathlib import Path
import time

def run_all_notebooks():
    """Run all notebooks in order."""
    notebooks_dir = Path("notebooks")
    
    # List of notebooks to run in sequence
    notebooks = [
        "01_eda.ipynb",
        "02_preprocessing.ipynb",
        "03_mining.ipynb",
        "04_modeling.ipynb",
        "05_anomaly_detection.ipynb",
        "06_evaluation_report.ipynb"
    ]
    
    print("=" * 60)
    print("STARTING BATCH EXECUTION OF NOTEBOOKS")
    print("=" * 60)
    
    start_total = time.time()
    
    for nb_name in notebooks:
        input_path = notebooks_dir / nb_name
        output_path = notebooks_dir / nb_name  # In-place execution or save to outputs/
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Executing: {nb_name}...")
        try:
            start_time = time.time()
            pm.execute_notebook(
                input_path=str(input_path),
                output_path=str(output_path),
                kernel_name='python3'
            )
            elapsed = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: {nb_name} (Took {elapsed:.2f}s)")
            
        except Exception as e:
            print(f"\n[{time.strftime('%H:%M:%S')}] ERROR in {nb_name}")
            print(f"Details: {str(e)}")
            print("\nAborting sequence due to error.")
            sys.exit(1)
            
    total_elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"ALL NOTEBOOKS EXECUTED SUCCESSFULLY IN {total_elapsed:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    # Change directory to project root if executed from scripts/
    if Path.cwd().name == "scripts":
        os.chdir("..")
        
    run_all_notebooks()
