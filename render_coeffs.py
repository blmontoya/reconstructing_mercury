"""
render_coeffs.py
Converts Spherical Harmonic Coefficients (.npz) into 2D Gravity Maps (.npy).
UPDATED: Correctly handles 'C_lm' and 'S_lm' keys.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pyshtools as pysh

def main():
    # --- CONFIGURATION ---
    input_path = "data/train/mercury/mercury_coeffs.npz"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    TARGET_SHAPE = (360, 720) 
    # ---------------------

    print(f"Loading {input_path}...")
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {input_path}")
        return

    try:
        data = np.load(input_path)
        keys = list(data.keys())
        
        # --- FIX STARTS HERE ---
        # 1. Check for the keys we saved ('C_lm' and 'S_lm')
        if 'C_lm' in keys and 'S_lm' in keys:
            c_coeffs = data['C_lm']
            s_coeffs = data['S_lm']
            # Stack them into shape (2, L+1, L+1) for pyshtools
            raw_data = np.array([c_coeffs, s_coeffs])
            
        # 2. Fallback for other formats
        elif 'C' in keys and 'S' in keys:
            c_coeffs = data['C']
            s_coeffs = data['S']
            raw_data = np.array([c_coeffs, s_coeffs])
        elif 'coeffs' in keys:
            raw_data = data['coeffs']
        else:
            print(f"Unknown keys in .npz: {keys}")
            return
        # --- FIX ENDS HERE ---

        # LOAD COEFFS INTO PYSHTOOLS
        # normalization='unnorm' is standard for these PDS files, 
        # but 'ortho' or '4pi' might be needed depending on exact science data.
        # usually 'ortho' is the safe default for pysh if uncertain.
        coeffs = pysh.SHCoeffs.from_array(raw_data)

    except Exception as e:
        print(f"Error loading file: {e}")
        # Print array shape to help debug
        try:
            print(f"Attempted to load array shape: {raw_data.shape}")
        except:
            pass
        return

    # ==========================================================
    # 1. GENERATE HIGH RES TARGET (L_MAX / Best Quality)
    # ==========================================================
    print(f"Generating High-Res Target (Using all L_max={coeffs.lmax})...")
    
    # Expand utilizing all available coefficients
    grid_high = coeffs.expand(grid='DH2') 
    gravity_high_raw = grid_high.data
    
    # Resize to target shape
    zoom_h = TARGET_SHAPE[0] / gravity_high_raw.shape[0]
    zoom_w = TARGET_SHAPE[1] / gravity_high_raw.shape[1]
    gravity_high_final = zoom(gravity_high_raw, (zoom_h, zoom_w), order=3)

    # Normalize High Res
    # (Optional: Adjust this threshold if your gravity values are naturally high)
    if np.max(np.abs(gravity_high_final)) > 800:
        gravity_high_final = (gravity_high_final - np.mean(gravity_high_final)) / np.std(gravity_high_final) * 80
    
    l50_path = os.path.join(output_dir, "mercury_grav_L50.npy")
    np.save(l50_path, gravity_high_final)
    print(f"SAVED High Res Target: {l50_path}")

    # ==========================================================
    # 2. GENERATE LOW RES INPUT (True L25 Truncation)
    # ==========================================================
    print("Generating Low-Res Input (Truncating to L=25)...")
    
    # This asks math to stop at L=25
    grid_low = coeffs.expand(grid='DH2', lmax=25) 
    gravity_low_raw = grid_low.data
    
    # Resize Low Res
    zoom_h_low = TARGET_SHAPE[0] / gravity_low_raw.shape[0]
    zoom_w_low = TARGET_SHAPE[1] / gravity_low_raw.shape[1]
    gravity_low_final = zoom(gravity_low_raw, (zoom_h_low, zoom_w_low), order=3)

    # Normalize Low Res
    if np.max(np.abs(gravity_low_final)) > 800:
        gravity_low_final = (gravity_low_final - np.mean(gravity_low_final)) / np.std(gravity_low_final) * 80

    l25_path = os.path.join(output_dir, "mercury_grav_L25.npy")
    np.save(l25_path, gravity_low_final)
    print(f"SAVED Low Res Input (True L25): {l25_path}")

    # ==========================================================
    # VERIFY VISUALLY
    # ==========================================================
    print("Visualizing...")
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2,1,1)
    plt.title("Input (L25) - Truncated")
    plt.imshow(gravity_low_final, cmap='RdBu_r')
    plt.colorbar()
    
    plt.subplot(2,1,2)
    plt.title(f"Target (L{coeffs.lmax}) - High Detail")
    plt.imshow(gravity_high_final, cmap='RdBu_r')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()