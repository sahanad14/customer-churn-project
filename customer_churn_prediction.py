import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

churn_counts = df['Churn'].value_counts()

plt.figure()
churn_counts.plot(kind='bar')
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(df[df['Churn'] == 'Yes']['MonthlyCharges'], alpha=0.5, label='Churn = Yes')
plt.hist(df[df['Churn'] == 'No']['MonthlyCharges'], alpha=0.5, label='Churn = No')
plt.xlabel("Monthly Charges")
plt.ylabel("Count")
plt.title("Monthly Charges vs Churn")
plt.legend()
plt.show()

plt.figure()
plt.hist(df[df['Churn'] == 'Yes']['tenure'], alpha=0.5, label='Churn = Yes')
plt.hist(df[df['Churn'] == 'No']['tenure'], alpha=0.5, label='Churn = No')
plt.xlabel("Tenure")
plt.ylabel("Count")
plt.title("Tenure vs Churn")
plt.legend()
plt.show()
