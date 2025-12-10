"""
Moon Pre-training + Mercury Fine-tuning
Runs the full paper's training strategy end-to-end
"""
import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and print its output"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: Command failed with return code {result.returncode}")
        return False

    print(f"\nSUCCESS: {description}")
    return True


def check_mercury_data_available():
    """Check if Mercury data is available for Phase 2"""
    mercury_files = [
        'data/processed/mercury_grav_L25.npy',
        'data/processed/mercury_grav_L50.npy',
        'data/processed/mercury_dem_720x360.npy'
    ]

    all_exist = all(os.path.exists(f) for f in mercury_files)

    if all_exist:
        return True
    else:
        print("\nMissing files:")
        for f in mercury_files:
            if not os.path.exists(f):
                print(f"  - {f}")
        return False


def main():
    """
    Complete two-phase workflow: Moon pre-training + Mercury fine-tuning

    Phase 1: Moon Pre-training
    - Preprocess DEM data
    - Train gravity-only baseline
    - Train full DEM-enhanced model
    - Compare performance

    Phase 2: Mercury Fine-tuning (if data available)
    - Fine-tune pre-trained model on Mercury
    - Compare zero-shot vs fine-tuned
    - Apply to southern hemisphere
    """
    print("="*80)
    print("COMPLETE TWO-PHASE WORKFLOW")
    print("Paper Implementation: Moon Pre-training + Mercury Fine-tuning")
    print("="*80)
    print("\nThis pipeline will run:")
    print("\nPHASE 1: MOON PRE-TRAINING")
    print("  1. Process Lunar DEM data to multiple resolutions")
    print("  2. Train gravity-only baseline model")
    print("  3. Train full model with DEM refining network")
    print("  4. Compare and evaluate both models")
    print("\nPHASE 2: MERCURY FINE-TUNING (if data available)")
    print("  5. Fine-tune Moon model on Mercury data")
    print("  6. Compare zero-shot vs fine-tuned performance")
    print("  7. Apply to Mercury southern hemisphere reconstruction")
    print("\nExpected total time:")
    print("  - Phase 1: 2-4 hours on CPU")
    print("  - Phase 2: 30-60 minutes on CPU (if Mercury data available)")
    print("="*80)

    response = input("\nProceed with complete workflow? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return

    # PHASE 1: MOON PRE-TRAINING
    print("\n" + "="*80)
    print("PHASE 1: MOON PRE-TRAINING")
    print("="*80)

    # Step 1: Preprocess DEM Data
    if not run_command(
        "python dem_preprocessing.py",
        "Step 1.1: Preprocess Lunar DEM data"
    ):
        return

    # Verify DEM files were created
    required_dems = [
        "data/processed/moon_dem_360x180.npy",
        "data/processed/moon_dem_720x360.npy",
        "data/processed/moon_dem_1440x720.npy",
        "data/processed/moon_dem_2880x1440.npy"
    ]

    print("\nVerifying DEM files...")
    for dem_file in required_dems:
        if os.path.exists(dem_file):
            print(f"  Found: {dem_file}")
        else:
            print("ERROR: DEM preprocessing did not complete successfully")
            return

    # Step 2: Train Gravity-Only Model (Baseline)
    if not run_command(
        "python train.py --l_low 25 --l_high 200 --epochs 100 --batch_size 32",
        "Step 1.2: Train gravity-only baseline model"
    ):
        return

    # Step 3: Train Full Model with DEM
    if not run_command(
        "python train_with_dem.py --l_low 25 --l_high 200 --epochs 100 --batch_size 32",
        "Step 1.3: Train full model with DEM refining network"
    ):
        return

    # Step 4: Evaluate and Compare Models
    print("\n" + "="*80)
    print("Step 1.4: Evaluating and comparing models...")
    print("="*80)

    if not run_command(
        "python evaluate_dem_model.py "
        "--gravity_only_model checkpoints/moon_gravity_model_best.h5 "
        "--model_path checkpoints_dem/moon_full_model_best.h5 "
        "--gravity_low data/processed/moon_grav_L25.npy "
        "--gravity_high data/processed/moon_grav_L200.npy "
        "--dem_high data/processed/moon_dem_2880x1440.npy "
        "--compare",
        "Step 1.4: Compare Moon models and generate evaluation metrics"
    ):
        return

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE!")
    print("="*80)
    print("\nGenerated outputs:")
    print("  DEM data: data/processed/moon_dem_*.npy")
    print("  ravity-only model: checkpoints/moon_gravity_model_best.h5")
    print("  Full DEM model: checkpoints_dem/moon_full_model_best.h5")
    print("  raining curves:")
    print("     - checkpoints/training_curves.png")
    print("     - checkpoints_dem/training_curves_dem.png")
    print("  Evaluation results:")
    print("     - results_gravity_only/full_comparison.png")
    print("     - results_gravity_dem/full_comparison.png")


    # PHASE 2: MERCURY FINE-TUNING 
    print("\n" + "="*80)
    print("PHASE 2: MERCURY FINE-TUNING")
    print("="*80)

    mercury_available = check_mercury_data_available()

    if not mercury_available:
        return

    response = input("\nMercury data found. Proceed with Phase 2 fine-tuning? (y/n): ")
    if response.lower() != 'y':
        print("Phase 2 skipped. Phase 1 complete.")
        print("\nTo run Phase 2 later:")
        print("  python train_mercury_finetuning.py --compare")
        return

    # Step 5: Fine-tune on Mercury
    if not run_command(
        "python train_mercury_finetuning.py "
        "--moon_model checkpoints_dem/moon_full_model_best.h5 "
        "--mercury_grav_low data/processed/mercury_grav_L25.npy "
        "--mercury_grav_high data/processed/mercury_grav_L50.npy "
        "--mercury_dem_high data/processed/mercury_dem_720x360.npy "
        "--l_low 25 --l_high 50 "
        "--epochs 50 --batch_size 16 --lr 1e-5 "
        "--compare",
        "Step 2.1: Fine-tune Moon model on Mercury data"
    ):
        return

    print("\n" + "="*80)
    print("  Mercury fine-tuned model: checkpoints_mercury/mercury_model_best.h5")
    print("  Fine-tuning curves: checkpoints_mercury/finetuning_log.csv")
    print("  Comparison results:")
    print("     - results_mercury_comparison/moon_zero_shot/")
    print("     - results_mercury_comparison/mercury_finetuned/")



if __name__ == "__main__":
    main()
