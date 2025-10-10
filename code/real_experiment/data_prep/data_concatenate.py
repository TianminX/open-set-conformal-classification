import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------
# File paths (adjust these paths as needed)
# -------------------------------
# npz_file_path = "../../data/celebrity/embeddings/batch1.npz"
npz_file_path = "combined_data.npz"
csv_file_path = "../../data/celebrity/list_attr_celeba.csv"  # CSV file with image_name and 40 binary attribute columns
output_file_path = "concatenated_combined_data.npz"

# -------------------------------
# Step 1: Load npz file data
# -------------------------------
data = np.load(npz_file_path)
X = data['X']            # Shape: (n_samples, 128)
Y = data['Y']            # Shape: (n_samples,)
img_names_npz = data['image_name']  # e.g., array of strings like ['000001.jpg', '000002.jpg', ...]

# -------------------------------
# Step 2: Convert npz data to a DataFrame for merging
# -------------------------------
# Create a DataFrame with the image names and embeddings
df_embeddings = pd.DataFrame(X, columns=[f'embed_{i}' for i in range(X.shape[1])])
df_embeddings['image_name'] = pd.Series(img_names_npz).str.strip()  # ensure no extra whitespace
df_embeddings['Y'] = Y

# -------------------------------
# Step 3: Load the CSV file containing attributes
# -------------------------------
df_attr = pd.read_csv(csv_file_path)
# Assuming the CSV file has a header with "image_name" as the first column and the remaining columns are attributes.
df_attr.rename(columns={'image_id': 'image_name'}, inplace=True)
df_attr['image_name'] = df_attr['image_name'].astype(str).str.strip()

# -------------------------------
# Step 4: Merge the DataFrames on image_name
# -------------------------------
df_merged = pd.merge(df_embeddings, df_attr, on='image_name', how='inner')
print(f"Merged dataset shape: {df_merged.shape}")

# -------------------------------
# Step 5: Normalize the 128-d embeddings
# -------------------------------
embedding_cols = [f'embed_{i}' for i in range(128)]
scaler = StandardScaler()
df_merged[embedding_cols] = scaler.fit_transform(df_merged[embedding_cols])

# -------------------------------
# Step 6: Process attribute columns
# -------------------------------
# Assuming all columns in df_attr except "image_name" are binary attributes.
attr_cols = [col for col in df_attr.columns if col != 'image_name']
# Convert attributes to integers (if they are not already)
df_merged[attr_cols] = df_merged[attr_cols].astype(int)

# -------------------------------
# Step 7: Concatenate the normalized embeddings with the binary attributes
# -------------------------------
concatenated_features = df_merged[embedding_cols + attr_cols].values
print("Concatenated features shape:", concatenated_features.shape)

# Convert image names to a NumPy array of strings (matching the original file type)
image_names = np.array(df_merged['image_name'].values, dtype=np.str_)

# -------------------------------
# Save the concatenated data to a new npz file
# -------------------------------
np.savez(output_file_path,
         X_concat=concatenated_features,
         Y=df_merged['Y'].values,
         image_name=image_names)
print("Concatenated data saved to:", output_file_path)
