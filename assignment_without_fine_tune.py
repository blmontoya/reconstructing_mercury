"""
Runs full workflow from data preprocessing to evaluation
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


def main():
    """
    Complete pipeline for DEM-enhanced gravity reconstruction

    Pipeline Steps:
    1. Preprocess DEM data from TIFF files
    2. Train gravity-only model (baseline)
    3. Train gravity+DEM model (full model)
    4. Evaluate and compare both models
    """

    response = input("\nProceed with full pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return

    # step 1: Preprocess DEM Data
    if not run_command(
        "python dem_preprocessing.py",
        "Preprocess Lunar DEM data"
    ):
        return

    # verify DEM files were created
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
            print(f"  MISSING: {dem_file}")
            print("ERROR: DEM preprocessing did not complete successfully")
            return

    # step 2: Train Gravity-Only Model 
    if not run_command(
        "python train.py --l_low 25 --l_high 200 --epochs 100 --batch_size 32",
        "Train gravity-only model (baseline)"
    ):
        return

    # step 3: Train Full Model with DEM
    if not run_command(
        "python train_with_dem.py --l_low 25 --l_high 200 --epochs 100 --batch_size 32",
        "Train full model with DEM refining network"
    ):
        return

    # step 4: Evaluate and Compare Models
    print("\n" + "="*80)
    print("Comparing Gravity-Only vs Gravity+DEM Models")


    if not run_command(
        "python evaluate_dem_model.py "
        "--gravity_only_model checkpoints/moon_gravity_model_best.h5 "
        "--model_path checkpoints_dem/moon_full_model_best.h5 "
        "--gravity_low data/processed/moon_grav_L25.npy "
        "--gravity_high data/processed/moon_grav_L200.npy "
        "--dem_high data/processed/moon_dem_2880x1440.npy "
        "--compare",
        "Compare models and generate evaluation metrics"
    ):
        return


    print("1. DEM data: data/processed/moon_dem_*.npy")
    print("2. Gravity-only model: checkpoints/moon_gravity_model_best.h5")
    print("3. Full DEM model: checkpoints_dem/moon_full_model_best.h5")
    print("4. Training curves:")
    print("   - checkpoints/training_curves.png")
    print("   - checkpoints_dem/training_curves_dem.png")
    print("5. Evaluation results:")
    print("   - results_gravity_only/full_comparison.png")
    print("   - results_gravity_dem/full_comparison.png")
    print("\nNext steps:")
    print("- Apply pre-trained model to Mercury data (fine-tuning)")


if __name__ == "__main__":
    main()
