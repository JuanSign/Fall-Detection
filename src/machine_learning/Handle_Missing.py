import os
import pandas as pd
from sklearn.impute import KNNImputer

root = "data/pose"
for file_name in os.listdir(root):
    if file_name.endswith((".csv")):
        print("HANDLING MISSING VALUE ON: " + file_name)
        data = pd.read_csv(os.path.join(root, file_name))

        # Temporary remove label & path
        dropped_data = data.drop(columns=["path"], errors="ignore")

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
        dropped_data[:] = imputer.fit_transform(dropped_data)

        # Append label & path
        final_data = pd.concat([dropped_data, data[["path"]]], axis=1)

        folder = os.path.join(root, "missing")
        with open(os.path.join(folder, file_name), "w") as file:
            # Tampilkan hasil imputer
            final_data.to_csv(file, index=False)
