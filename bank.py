# bank_marketing_decision_tree.py

import pandas as pd
import zipfile
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Download and Extract Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
zip_path = "bank.zip"
data_folder = "bank-additional"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

if not os.path.exists(data_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

# Step 2: Load Dataset
df = pd.read_csv("bank-additional/bank-additional-full.csv", sep=';')
print("Dataset loaded. Shape:", df.shape)

# Step 3: Encode Categorical Variables
df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

# Step 4: Split Data
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 5: Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = clf.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Visualize Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, 
               feature_names=X.columns, 
               class_names=label_encoders['y'].classes_, 
               filled=True,
               rounded=True,
               fontsize=10)
plt.title("Decision Tree for Bank Marketing")
plt.tight_layout()
plt.savefig("decision_tree.png")  # Saves the plot as a PNG
plt.show()
