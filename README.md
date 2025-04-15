import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("sider_dataset.csv")
data['side_effects'] = data['side_effects'].apply(lambda x: x.split('|'))

# Feature: Drug ID or molecular representation (simplified here)
X = pd.get_dummies(data['drug_name'])
y = data['side_effects']

# Convert side effects into binary labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
