# SAMPLING ASSIGNMENT
# Author: Kamya_Mehra_102213025_3C75
# Date: 25 January 2025

import pandas as pd
data = pd.read_csv("Creditcard_data.csv")
data.head(), data.info(), data.describe()

# Checking class distribution
class_distribution = data['Class'].value_counts(normalize=True) * 100
print(class_distribution)



from imblearn.over_sampling import SMOTE
# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Check the new class distribution
balanced_class_distribution = y_balanced.value_counts(normalize=True) * 100
print(balanced_class_distribution)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Split the balanced dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42,
                                                    stratify=y_balanced)

# Define sampling techniques
def random_sampling(X, y, sample_size):
    indices = np.random.choice(range(len(X)), size=sample_size, replace=False)
    return X.iloc[indices], y.iloc[indices]

def stratified_sampling(X, y, sample_size):
    return train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)

# Sampling sizes (we can adjust them based on our sample size detection formula)
sample_sizes = [100, 200, 300, 400, 500]

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to store results
results = {}

# Apply sampling and train models
for i, sample_size in enumerate(sample_sizes):
    results[f"Sample {i + 1}"] = {}

    # Apply sampling techniques
    X_sample, y_sample = random_sampling(X_train, y_train, sample_size)

    for model_name, model in models.items():
        # Train the model
        model.fit(X_sample, y_sample)

        # Test the model
        y_pred = model.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[f"Sample {i + 1}"][model_name] = accuracy

# Display results
results_df = pd.DataFrame(results)
print(results_df)
