import os
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


# Daftar fitur yang ingin digunakan
features = [
    "NOSE_X", "NOSE_Y", "LEFT_EYE_INNER_X", "LEFT_EYE_INNER_Y", "LEFT_EYE_X", "LEFT_EYE_Y", 
    "LEFT_EYE_OUTER_X", "LEFT_EYE_OUTER_Y", "RIGHT_EYE_INNER_X", "RIGHT_EYE_INNER_Y", "RIGHT_EYE_X", 
    "RIGHT_EYE_Y", "RIGHT_EYE_OUTER_X", "RIGHT_EYE_OUTER_Y", "LEFT_EAR_X", "LEFT_EAR_Y", "RIGHT_EAR_X", 
    "RIGHT_EAR_Y", "MOUTH_LEFT_X", "MOUTH_LEFT_Y", "MOUTH_RIGHT_X", "MOUTH_RIGHT_Y", "LEFT_SHOULDER_X", 
    "LEFT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y", "LEFT_HIP_X", "LEFT_HIP_Y", "RIGHT_HIP_X", 
    "RIGHT_HIP_Y"
]

root = "data/pose"
all_data = []

# Menggabungkan data dari berbagai file CSV
for file_name in os.listdir(root):
    if file_name.endswith('.csv'):
        data = pd.read_csv(os.path.join(root, file_name))
        all_data.append(data)

# Menggabungkan semua data menjadi satu dataframe
combined_data = pd.concat(all_data, ignore_index=True)

# Menyaring data untuk hanya mencakup fitur yang diinginkan
y = combined_data['label']  # Pastikan kolom target ada di dataset Anda
combined_data = combined_data[features]

# Menangani missing values menggunakan SimpleImputer
imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)

# imputer = SimpleImputer(strategy='mean')  # Menggunakan rata-rata untuk menggantikan missing values
combined_data_imputed = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns)

# Membagi data menjadi fitur (X) dan target (y)
# Misalkan kolom targetnya bernama 'target', ganti dengan nama target yang sesuai
X = combined_data_imputed

# Membagi data training menjadi training dan validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Membaca data test dari file CSV terpisah
test_data = pd.read_csv("data/test/test_data.csv")  # Ganti dengan path file data test Anda
path_test = test_data['path']
img_name_test = [path.split("\\")[-1] for path in path_test]

# Menyaring dan menangani missing values pada data test
test_data_filtered = test_data[features]
test_data_imputed = pd.DataFrame(imputer.transform(test_data_filtered), columns=test_data_filtered.columns)
X_test_scaled = scaler.transform(test_data_imputed)  # Melakukan transformasi pada data test

# Melatih model Random Forest
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train_scaled, y_train)

# K-Fold Cross Validation pada data training
cv_scores = cross_val_score(RF, X_train_scaled, y_train, cv=5)  # K-Fold cross-validation dengan 5 folds
print(f'K-Fold Cross Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

# Evaluasi model menggunakan data validation
y_val_pred = RF.predict(X_val_scaled)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Evaluasi model menggunakan data test
y_test_pred = RF.predict(X_test_scaled)
print("Test Classification Report:")
print(y_test_pred)
# print(classification_report(y_test, y_test_pred))

with open("submission.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows([["id", "label"]])
    for index, name in enumerate(img_name_test):
        writer.writerow([name, y_test_pred[index]])