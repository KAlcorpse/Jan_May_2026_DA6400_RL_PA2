import os
import shutil

SENS_RHO       = [1, 4]
BATCH_SIZES    = [16, 32, 64, 128, 256]
TARGET_UPDATES = [100, 250, 500, 1000, 2000]
SENS_SEEDS     = list(range(10))

def organize_file(filename, rho, subfolder):
    """Helper function to create directories and copy the file."""
    if os.path.exists(filename):
        # Creates paths like: rho_1/batch or rho_4/target
        dir_path = os.path.join(f"rho_{rho}", subfolder)
        
        # Ensure the directory exists (equivalent to mkdir -p)
        os.makedirs(dir_path, exist_ok=True)
        
        # Construct the final destination path
        destination = os.path.join(dir_path, filename)
        
        # Copy the file (change to shutil.move if you want to delete the originals)
        shutil.copy(filename, destination)
    else:
        print(f" Not found → {filename}")

print("── Processing Batch folder: tu=500 fixed, bs varies ──")
for rho in SENS_RHO:
    for bs in BATCH_SIZES:
        for seed in SENS_SEEDS:
            filename = f"sens_rho{rho}_bs{bs}_tu500_seed{seed}.pkl"
            organize_file(filename, rho, "batch")

print("\n── Processing Target folder: bs=64 fixed, tu varies ──")
for rho in SENS_RHO:
    for tu in TARGET_UPDATES:
        for seed in SENS_SEEDS:
            filename = f"sens_rho{rho}_bs64_tu{tu}_seed{seed}.pkl"
            organize_file(filename, rho, "target")

print("\n All done!")