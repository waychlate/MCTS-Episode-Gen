from pathlib import Path
import random

# 1. Define your paths
SOURCE_DATA_DIR = Path("/blue/iruchkin/khek.do/output_ttc")
DESTINATION_DATA_DIR = Path("/blue/iruchkin/khek.do/output_ttc_sorted")

TRAIN_DIR = DESTINATION_DATA_DIR / "train"
TEST_DIR = DESTINATION_DATA_DIR / "test"

# Create the destination directories if they don't exist
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Gather all unique completed episode IDs
csv_files = SOURCE_DATA_DIR.glob("episode_*_data.csv")
episode_ids = sorted([int(f.name.split('_')[1]) for f in csv_files])

if not episode_ids:
    print(f"No completed episode files found in {SOURCE_DATA_DIR}!")
    exit()

# Shuffle with a fixed seed so the split is mathematically reproducible
random.seed(42)
random.shuffle(episode_ids)

# Determine split indices
split_idx = int(0.9 * len(episode_ids))
train_ids = set(episode_ids[:split_idx])
test_ids = set(episode_ids[split_idx:])

print(f"Found {len(episode_ids)} total episodes.")
print(f"Sorting {len(train_ids)} into train/ and {len(test_ids)} into test/...")

# Move the paired files cleanly
moved_count = 0

for ep_id in episode_ids:
    print(f"Sorted {moved_count} episodes, currently working on episodes {ep_id:04d}")

    # Determine the destination folder
    dest_folder = TRAIN_DIR if ep_id in train_ids else TEST_DIR
    
    # Define filenames for both parts of the episode
    csv_name = f"episode_{ep_id:04d}_data.csv"
    npz_name = f"episode_{ep_id:04d}_visuals.npz"
    
    # Move CSV file
    src_csv = SOURCE_DATA_DIR / csv_name
    dst_csv = dest_folder / csv_name
    if src_csv.exists():
        src_csv.rename(dst_csv)
        
    # Move NPZ file
    src_npz = SOURCE_DATA_DIR / npz_name
    dst_npz = dest_folder / npz_name
    if src_npz.exists():
        src_npz.rename(dst_npz)
        
    moved_count += 1

print(f"Successfully organized {moved_count} episodes into separate folders!")