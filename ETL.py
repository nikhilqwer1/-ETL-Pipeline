# etl_pipeline.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Step 1: Extract (Load Dataset)
# -----------------------------
def extract_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Extracted. Shape:", df.shape)
    return df

# -----------------------------
# Step 2: Transform
# -----------------------------
def transform_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    df[['Age', 'Embarked']] = imputer.fit_transform(df[['Age', 'Embarked']])

    # Encode categorical columns
    label_encoders = {}
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature scaling
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    print("Data Transformed.")
    return df

# -----------------------------
# Step 3: Load (Save Clean Data)
# -----------------------------
def load_data(df, output_file="processed_data.csv"):
    df.to_csv(output_file, index=False)
    print(f"Data loaded to {output_file}")

# -----------------------------
# Main ETL Execution
# -----------------------------
if __name__ == "__main__":
    # Use your actual dataset here
    input_file = "titanic.csv"  # replace with your path
    df_raw = extract_data(input_file)
    df_clean = transform_data(df_raw)
    load_data(df_clean)
