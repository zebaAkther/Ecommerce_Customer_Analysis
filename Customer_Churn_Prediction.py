import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("churn prediction.csv")

# -------------------------------
# 1. BASIC INFO
# -------------------------------
print("Initial Shape:", df.shape)
print(df.info())

# -------------------------------
# 2. REMOVE DUPLICATES
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# 3. HANDLE MISSING VALUES
# -------------------------------

# Numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -------------------------------
# 4. FIX INCONSISTENT DATA
# -------------------------------

# Fix Gender
df['Gender'] = df['Gender'].replace({
    'M': 'Male',
    'F': 'Female'
})

# Fix Yes/No values
df = df.replace({
    'yes': 'Yes',
    'no': 'No',
    'Y': 'Yes',
    'N': 'No',
    True: 'Yes',
    False: 'No'
})

# Fix Country format
df['Country'] = df['Country'].str.strip().str.title()

# -------------------------------
# 5. DATE CONVERSION
# -------------------------------
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')

df['Year'] = df['LastPurchaseDate'].dt.year
df['Month'] = df['LastPurchaseDate'].dt.month

df.drop('LastPurchaseDate', axis=1, inplace=True)

# -------------------------------
# 6. DROP USELESS COLUMN
# -------------------------------
df.drop('CustomerID', axis=1, inplace=True)

# -------------------------------
# 7. ENCODE TARGET VARIABLE
# -------------------------------
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# -------------------------------
# 8. ENCODE CATEGORICAL VARIABLES
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)   # IMPORTANT FIX
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 9. FINAL CHECK
# -------------------------------
print("\nFinal Shape:", df.shape)
print("\nNull Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

print("\nCleaned Data Sample:\n", df.head())

# -------------------------------
# 10. SAVE CLEAN DATA (OPTIONAL)
# -------------------------------
df.to_csv("cleaned_churn_data.csv", index=False)

# Fill missing Year and Month
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Month'] = df['Month'].fillna(df['Month'].median())

print(df.isnull().sum())

df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)

print("\n✅ Data cleaning completed successfully!")

# -------------------------------
# 11. EDA (EXPLORATORY DATA ANALYSIS)
# -------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. WHO CHURNS MORE?
# -------------------------------
print("\nChurn Count:\n", df['Churn'].value_counts())

sns.countplot(x='Churn', data=df)
plt.title("Who Churns More (0 = No, 1 = Yes)")
plt.show()


# -------------------------------
# 2. INCOME VS CHURN
# -------------------------------
plt.figure()
sns.boxplot(x='Churn', y='Income', data=df)
plt.title("Income vs Churn")
plt.show()


# -------------------------------
# 3. SPENDING SCORE VS CHURN
# -------------------------------
plt.figure()
sns.boxplot(x='Churn', y='SpendingScore', data=df)
plt.title("Spending Score vs Churn")
plt.show()

# -------------------------------
# 12. MODEL BUILDING
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X = df[['Age', 'Income', 'SpendingScore', 'PurchaseAmount', 'ReviewScore', 'SessionTime']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for KNN, SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 13. APPLY MODELS
# -------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    # -------------------------------
# 14. RESULTS COMPARISON
# -------------------------------

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print("\nModel Comparison:\n")
print(results_df)

# -------------------------------
# 15. EVALUATION METRICS (FINAL)
# -------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train best model
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train, y_train)

# Predict
y_pred = best_model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import pickle


# Save model + scaler
pickle.dump(best_model, open("churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model saved successfully")