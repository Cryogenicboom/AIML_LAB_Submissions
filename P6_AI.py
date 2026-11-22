import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

df.head()
# 
df.info()
display(df.head())

display(df.describe())

#

print("Missing values before preprocessing:")
print(df.isnull().sum())

# 

df_clean = df.copy()

df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)

df_clean["Embarked"] = df_clean["Embarked"].fillna(
    df_clean["Embarked"].mode()[0]
)

df_clean = df_clean.drop(columns=["Cabin"])

print("Missing values after handling:")
print(df_clean.isnull().sum())

# 

le = LabelEncoder()
df_clean["Sex"] = le.fit_transform(df_clean["Sex"])

print("Sex encoding:")
print(le.classes_)

df_clean = pd.get_dummies(
    df_clean,
    columns=["Embarked"],
    dtype=int
)

display(df_clean.head())

# 

scaler = MinMaxScaler()

# Normalize Age and Fare
df_clean[["Age", "Fare"]] = scaler.fit_transform(
    df_clean[["Age", "Fare"]]
)

display(df_clean[["Age", "Fare"]].head())

print(df_clean[["Age", "Fare"]].min())
print(df_clean[["Age", "Fare"]].max())

# 

print("Final dataset shape:", df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum().sum())

display(df_clean.head())

df_clean.to_csv("titanic_preprocessed.csv", index=False)

print("Preprocessed dataset saved successfully!")


