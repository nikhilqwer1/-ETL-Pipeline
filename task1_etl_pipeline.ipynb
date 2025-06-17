import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Extract
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Transform
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# Load
df.to_csv("cleaned_titanic.csv", index=False)
print(df.head())
