"""
Train DEM Prediction Model Only
Predicts DEM from low-resolution gravity data

This is Step 2 of 3:
  Step 1 (DONE): Gravity model (low-res grav → high-res grav)
  Step 2 (THIS): DEM model (low-res grav → DEM)
  Step 3 (TODO): Merge models (grav + DEM → final reconstruction)
"""
import os
import sys
import subprocess


def check_prerequisites():
    """Check if we have everything needed to train DEM model"""
    print("="*80)
    print("CHECKING PREREQUISITES")
    print("="*80)
    
    issues = []
    
    # Check for preprocessed DEM data
    print("\n1. Checking DEM data...")
    dem_files = {
        'Moon DEM L25': 'data/processed/moon_dem_L25.npz',
        'Moon DEM L50': 'data/processed/moon_dem_L50.npz',
        'Moon DEM L200': 'data/processed/moon_dem_L200.npz',
    }
    
    missing_dem = []
    for name, path in dem_files.items():
        if os.path.exists(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - NOT FOUND")
            missing_dem.append(path)
    
    if missing_dem:
        issues.append("DEM data not preprocessed")
        print("\n  → Run: python preprocess_dem_synthetic.py")
    
    # Check for gravity data (needed as input)
    print("\n2. Checking gravity data (needed as input)...")
    grav_files = {
        'Moon Gravity L25': 'data/processed/moon_grav_L25.npz',
        'Moon Gravity L200': 'data/processed/moon_grav_L200.npz',
    }
    
    missing_grav = []
    for name, path in grav_files.items():
        if os.path.exists(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - NOT FOUND")
            missing_grav.append(path)
    
    if missing_grav:
        issues.append("Gravity data not preprocessed")
        print("\n  → Run: python preprocess_gravity.py")
    
    # Check for training script
    print("\n3. Checking for DEM training script...")
    if os.path.exists("train_stage2_dem.py"):
        print("  ✓ train_stage2_dem.py found")
    elif os.path.exists("train_with_dem.py"):
        print("  ✓ train_with_dem.py found")
    else:
        print("  ✗ No DEM training script found")
        issues.append("Missing training script")
        print("\n  → Need either train_stage2_dem.py or train_with_dem.py")
    
    # Check trained gravity model (optional - for reference)
    print("\n4. Checking for trained gravity model (optional)...")
    if os.path.exists("checkpoints/moon_gravity_model_best.h5"):
        print("  ✓ Gravity model found (for future merging)")
    else:
        print("  ⚠ Gravity model not found")
        print("    (Not required now, but needed for final merging step)")
    
    # Summary
    print("\n" + "="*80)
    if issues:
        print("❌ PREREQUISITES NOT MET")
        print("="*80)
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("✅ ALL PREREQUISITES MET")
        print("="*80)
        return True


def train_dem_model():
    """Train the DEM prediction model"""
    print("\n" + "="*80)
    print("TRAINING DEM PREDICTION MODEL")
    print("="*80)
    print("\nThis model learns: Low-Resolution Gravity → DEM")
    print("\nTraining configuration:")
    print("  Input: Moon gravity L25 (low resolution)")
    print("  Output: Moon DEM L200 (high resolution)")
    print("  Architecture: CNN with upsampling")
    print("  Expected time: 1-3 hours on CPU, 15-30 min on GPU")
    print("="*80)
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return False
    
    # Determine which script to use
    if os.path.exists("train_stage2_dem.py"):
        cmd = "python train_stage2_dem.py"
        script_name = "train_stage2_dem.py"
    else:
        cmd = "python train_with_dem.py --l_low 25 --l_high 200 --epochs 100 --batch_size 32"
        script_name = "train_with_dem.py"
    
    print(f"\nRunning: {cmd}\n")
    print("="*80)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED")
        print("="*80)
        return False
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    return True


def check_trained_model():
    """Check if DEM model was trained successfully"""
    print("\n" + "="*80)
    print("CHECKING TRAINED MODEL")
    print("="*80)
    
    # Look for model in common locations
    possible_locations = [
        "checkpoints_dem/moon_dem_model_best.h5",
        "checkpoints_dem/best_model.h5",
        "checkpoints/dem/moon_dem_model_best.h5",
        "checkpoints/dem/best_model.h5",
        "models/dem_model.h5",
    ]
    
    found_model = None
    for path in possible_locations:
        if os.path.exists(path):
            found_model = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"\n✓ Model found: {path}")
            print(f"  Size: {size_mb:.2f} MB")
            break
    
    if not found_model:
        print("\n⚠ Could not find trained model in expected locations")
        print("  Checked:")
        for path in possible_locations:
            print(f"    - {path}")
        return None
    
    # Check for training logs
    log_locations = [
        "checkpoints_dem/training_log.csv",
        "checkpoints_dem/history.csv",
        "logs/dem_training.log",
    ]
    
    for log_path in log_locations:
        if os.path.exists(log_path):
            print(f"✓ Training log: {log_path}")
            break
    
    return found_model


def main():
    """
    Train DEM prediction model
    This is Step 2 of the 3-step process
    """
    print("="*80)
    print("TRAIN DEM PREDICTION MODEL")
    print("Step 2 of 3: Gravity → DEM")
    print("="*80)
    print("\n📋 PIPELINE OVERVIEW:")
    print("  Step 1: ✅ Gravity model (low-res → high-res gravity)")
    print("  Step 2: 🔄 DEM model (low-res gravity → DEM) ← YOU ARE HERE")
    print("  Step 3: ⏭️ Merge models (gravity + DEM → final)")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Cannot proceed. Please fix issues above.")
        return
    
    print("\n" + "="*80)
    print("Ready to train DEM model!")
    print("="*80)
    
    # Train the model
    if not train_dem_model():
        return
    
    # Verify model was created
    model_path = check_trained_model()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if model_path:
        print("\n✅ DEM model training complete!")
        print(f"\nModel saved at: {model_path}")
        print("\n📋 What you have now:")
        print("  1. ✅ Gravity model: checkpoints/moon_gravity_model_best.h5")
        print(f"  2. ✅ DEM model: {model_path}")
        print("\n🎯 Next step (Step 3):")
        print("  Merge both models to create final high-resolution reconstruction")
        print("\n  Options for merging:")
        print("    A) Simple ensemble: average predictions from both models")
        print("    B) Weighted ensemble: learned weights for each model")
        print("    C) Fusion network: train a small network to combine outputs")
        print("\n  Create a script like: python merge_models.py")
    else:
        print("\n⚠ Model training completed but location is unclear")
        print("  Check your training script output for model save location")
    
    print("\n" + "="*80)
    print("For Mercury application:")
    print("  1. Train DEM model on Moon (DONE)")
    print("  2. Fine-tune on Mercury north pole data")
    print("  3. Apply merged model to Mercury south pole")
    print("="*80)


if __name__ == "__main__":
    main()