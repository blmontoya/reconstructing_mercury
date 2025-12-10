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
        print(f"\n❌ ERROR: File not found: {script_name}")
        print("Please ensure you have renamed your files according to the numbered list.")
        return False

    cmd = f"{sys.executable} {script_name} {args}"
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    
    print(f"\n✅ SUCCESS: {description}")
    print(f"Time taken: {duration/60:.2f} minutes")
    return True

def main():
    print("="*80)
    print("🚀 STARTING GRAVITY RECONSTRUCTION PIPELINE")
    print("="*80)
    print("Plan:")
    print(" 1. Preprocess Moon Gravity")
    print(" 2. Preprocess Moon DEM")
    print(" 3. Train Base Model on Moon (Dual-Stream)")
    print(" 4. Preprocess Mercury Data")
    print(" 5. Fine-Tune on Mercury North Pole")
    print(" 6. Reconstruct Mercury South Pole")
    print("="*80)

    # Confirm start
    # input("\nPress Enter to start the pipeline (or Ctrl+C to cancel)...")

    # --- PHASE 1: MOON PREPARATION ---
    if not run_step("step1_preprocess_moon_grav.py", "Step 1: Process Moon Gravity Data"):
        return

    if not run_step("step2_preprocess_moon_dem.py", 
                    "Step 2: Process Moon DEM Data",
                    args="--target_shapes 402 804 1440 2880"): # Ensures we generate L200 size
        return

    # --- PHASE 2: MOON TRAINING ---
    # Training takes the longest. We use 50 epochs as a safe default.
    if not run_step("step3_train_moon_base.py", 
                    "Step 3: Train Base Model on Moon",
                    args="--epochs 50 --batch_size 32 --l_high 200"):
        return

    # --- PHASE 3: MERCURY PREPARATION ---
    if not run_step("step4_preprocess_mercury.py", "Step 4: Process Mercury Data"):
        return

    # --- PHASE 4: MERCURY FINE-TUNING ---
    # We fine-tune on the North Pole
    if not run_step("step5_finetune_and_reconstruct.py", 
                    "Step 5: Fine-Tune on Mercury North",
                    args="--epochs 50 --lr 1e-5"):
        return

    # --- PHASE 5: FINAL RECONSTRUCTION ---
    # Apply the model to the South Pole
    if not run_step("step6_reconstruct_mercury.py", 
                    "Step 6: Reconstruct Mercury South Pole"):
        return

    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print("Check your final results in:")
    print("  results_reconstruction/mercury_reconstruction_comparison_new.pdf")
    print("  results_reconstruction/mercury_final_map_new.npy")

if __name__ == "__main__":
    main()