import os
import glob
import random

# 1. Define your paths
DATA_DIR = "output"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Create the target folders if they don't exist yet
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 2. Gather all unique completed episode IDs
csv_files = glob.glob(os.path.join(DATA_DIR, "episode_*_data.csv"))
episode_ids = sorted([int(os.path.basename(f).split('_')[1]) for f in csv_files])

if not episode_ids:
    print(f"No completed episode files found in {DATA_DIR}!")
    exit()

# 3. Shuffle with a fixed seed so the split is mathematically reproducible
random.seed(42)
random.shuffle(episode_ids)

# 4. Determine split indices (80% train, 20% test)
split_idx = int(0.9 * len(episode_ids))
train_ids = set(episode_ids[:split_idx])
test_ids = set(episode_ids[split_idx:])

print(f"Found {len(episode_ids)} total episodes.")
print(f"Sorting {len(train_ids)} into train/ and {len(test_ids)} into test/...")

# 5. Move the paired files cleanly
moved_count = 0

for ep_id in episode_ids:
    # Determine the destination folder
    dest_folder = TRAIN_DIR if ep_id in train_ids else TEST_DIR
    
    # Define filenames for both parts of the episode
    csv_name = f"episode_{ep_id:04d}_data.csv"
    npz_name = f"episode_{ep_id:04d}_visuals.npz"
    
    # Move CSV file
    src_csv = os.path.join(DATA_DIR, csv_name)
    dst_csv = os.path.join(dest_folder, csv_name)
    if os.path.exists(src_csv):
        os.rename(src_csv, dst_csv)
        
    # Move NPZ file
    src_npz = os.path.join(DATA_DIR, npz_name)
    dst_npz = os.path.join(dest_folder, npz_name)
    if os.path.exists(src_npz):
        os.rename(src_npz, dst_npz)
        
    moved_count += 1

print(f"Successfully organized {moved_count} episodes into separate folders!")