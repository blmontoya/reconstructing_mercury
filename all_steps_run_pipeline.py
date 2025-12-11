"""
MASTER PIPELINE: Moon Pre-training + Mercury Reconstruction
Runs the complete scientific workflow from raw data to final South Pole map.
"""
import os
import subprocess
import sys
import time

def run_step(script_name, description, args=""):
    """Runs a single python script and checks for errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"Script:  {script_name}")
    print("="*80)

    if not os.path.exists(script_name):
        print(f"\nERROR: File not found: {script_name}")
        print("Please ensure you have renamed your files according to the numbered list.")
        return False

    cmd = f"{sys.executable} {script_name} {args}"
    print(f"Command: {cmd}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"\nFAILED: {description}")
        return False

    print(f"\nSUCCESS: {description}")
    print(f"Time taken: {duration/60:.2f} minutes")
    return True

def main():
    # phase 1: moon preparation
    if not run_step("step1_preprocess_moon_gravity.py", "Step 1: Process Moon Gravity Data"):
        return

    if not run_step("step2_preprocess_moon_dem.py",
                    "Step 2: Process Moon DEM Data",
                    args="--target_shapes 402 804 1440 2880"):
        return

    # phase 2: moon training
    if not run_step("step3_train_moon_base.py",
                    "Step 3: Train Base Model on Moon",
                    args="--epochs 50 --batch_size 32 --l_high 200"):
        return

    # phase 3: mercury preparation
    if not run_step("step4_preprocess_mercury.py", "Step 4: Process Mercury Data"):
        return

    # phase 4: mercury fine-tuning
    step5_args = (
        "--moon_model checkpoints_dem/moon_full_model_best.h5 "
        "--mercury_grav_low data/processed/mercury_grav_L25.npy "
        "--mercury_grav_high data/processed/mercury_grav_L50.npy "
        "--mercury_dem_high data/processed/mercury_dem_720x360.npy "
        "--epochs 50 --lr 1e-5"
    )

    if not run_step("step5_finetune_and_reconstruct.py",
                    "Step 5: Fine-Tune on Mercury North",
                    args=step5_args):
        return

    # phase 5: final reconstruction
    step6_args = (
        "--model_path checkpoints_mercury/mercury_model_best.h5 "
        "--grav_low data/processed/mercury_grav_L25.npy "
        "--grav_high_truth data/processed/mercury_grav_L100.npy "
        "--dem_high data/processed/mercury_dem_L200.npy"
    )

    if not run_step("step6_reconstruct_mercury.py",
                    "Step 6: Reconstruct Mercury South Pole",
                    args=step6_args):
        return


    print("Check final results in:")
    print("  results_reconstruction/mercury_reconstruction_comparison_new.pdf")
    print("  results_reconstruction/mercury_final_map_new.npy")

if __name__ == "__main__":
    main()