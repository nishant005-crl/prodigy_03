import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("customer_data.csv")

# Encode categorical data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Feature selection
X = df[['Age', 'Gender', 'EstimatedSalary']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=['Age', 'Gender', 'EstimatedSalary'], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Purchase Prediction")
plt.show()
