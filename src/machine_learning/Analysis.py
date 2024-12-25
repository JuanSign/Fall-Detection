import os
import pandas as pd

root = "data\pose"
for file_name in os.listdir(root):
    if file_name.endswith(('.csv')):
        data = pd.read_csv(os.path.join(root, file_name))

        # Hitung jumlah missing value di setiap atribut
        missing_values = data.isnull().sum()

        folder = os.path.join(root, "metadata")
        with open (os.path.join(folder, file_name+".txt"), "w") as file:
            # Tampilkan hasil
            file.write("Jumlah missing values pada setiap atribut:\n")
            file.write(missing_values.to_string())
