import numpy as np

# Load the output npz file
output_file_path = "concatenated_combined_data.npz"
data = np.load(output_file_path)

# Print the keys contained in the file
print("Keys in the npz file:", data.files)

# Check the shape of each array
print("Concatenated features shape:", data['X_concat'].shape)
print("Labels shape:", data['Y'].shape)
print("Image names shape:", data['image_name'].shape)

# # Print sample data for verification
# print("\nFirst entry of concatenated features:\n", data['X_concat'][0])
# print("First label:", data['Y'][0])
# print("First image name:", data['image_name'][0])

for key in data.files:
    print(f"{key}: Shape = {data[key].shape}")
    print(f"Sample Data ({key}): {data[key][:1]}")