from pathlib import Path
import torch
import numpy as np
import torchvision.transforms.functional as TF

input_dir = Path('/blue/iruchkin/khek.do/sorted_output/train')
# input_dir = Path('/blue/iruchkin/khek.do/sorted_output/test')
output_dir = Path('/blue/iruchkin/khek.do/processed_output/processed_visual_train')
# output_dir = Path('/blue/iruchkin/khek.do/processed_output/processed_visual_test')

output_dir.mkdir(parents=True, exist_ok=True)

# Find all visual files matching the pattern
visual_files = sorted(input_dir.glob('*_visuals.npz'))
total_files = len(visual_files)

print(f"Found {total_files} episodes to process sequentially.")

for idx, visual_path in enumerate(visual_files):
    filename = visual_path.name

    print(f"[{idx+1}/{total_files}] Preprocessing {filename}")

    # Load raw visuals
    raw_data = np.load(visual_path)
    visual_array = raw_data['visuals']

    # Convert to PyTorch Tensor and permute to (Time, Channels, Height, Width) and normalize pixels to [0.0, 1.0]
    frames = torch.from_numpy(visual_array).float() / 255.0
    frames = frames.permute(0, 3, 1, 2)

    processed_frames = []

    for t in range(frames.shape[0]):
        frame = frames[t]
        cropped_frame = TF.crop(frame, top=60, left=0, height=90, width=600) 
        squeezed_frame = TF.resize(cropped_frame, size=[48, 320], antialias=True) 
        processed_frames.append(squeezed_frame)

    final_dataset_tensor = torch.stack(processed_frames)
    
    # Reverse layout, unnormalize, and cast back to dense uint8 for disk compression
    final_visuals_numpy = final_dataset_tensor.permute(0, 2, 3, 1).numpy() * 255.0

    # Save file as compressed npz
    output_path = output_dir / filename
    np.savez_compressed(output_path, visuals=final_visuals_numpy.astype(np.uint8))

print(f"Successfully processed and saved to: {output_path}. Ready for DIAMOND.")