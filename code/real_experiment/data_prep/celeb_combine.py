import os
import numpy as np

# Path to the directory containing .npz files
embedding_dir = "../../data/celebrity/embeddings/"

# List all .npz files in the directory
embedding_files = [os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) if f.endswith('.npz')]

print(f"Found {len(embedding_files)} embedding files.")

# Initialize containers to store combined data
all_X = []
all_Y = []
all_image_names = []

# Process each file
for file_path in embedding_files:
    print(f"Loading file: {file_path}")
    data = np.load(file_path)

    # Ensure the required keys are present
    if all(key in data for key in ['X', 'Y', 'image_name']):
        # Append data to the combined lists
        all_X.append(data['X'])
        all_Y.append(data['Y'])
        all_image_names.append(data['image_name'])
    else:
        print(f"Missing expected arrays in file: {file_path}")

# Combine data from all files
all_X = np.vstack(all_X)  # Combine along the first axis
all_Y = np.concatenate(all_Y)
all_image_names = np.concatenate(all_image_names)

# # Print summary of the combined data
# print(f"Combined X: Shape = {all_X.shape}")
# print(f"Combined Y: Shape = {all_Y.shape}")
# print(f"Combined image names: Shape = {all_image_names.shape}")
#
# # Example: Inspect first few entries
# print("Sample Data:")
# print(f"X[0]: {all_X[0]}")
# print(f"Y[0]: {all_Y[0]}")
# print(f"Image Name[0]: {all_image_names[0]}")

# Combine and save
np.savez_compressed("combined_data.npz", X=all_X, Y=all_Y, image_name=all_image_names)

# Load later
data = np.load("combined_data.npz")
X, Y, image_names = data['X'], data['Y'], data['image_name']