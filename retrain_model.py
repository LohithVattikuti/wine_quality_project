import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Load the wine dataset for retraining (replace with your dataset if different)
data = load_wine()
X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base learners (remove xgboost)
base_learners = [
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier())
]

# Create a stacking classifier
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression()
)

# Create a pipeline to standardize the features and train the stacked model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stacked_model', stacked_model)
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Test the accuracy (optional)
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save the retrained stacked model (without xgboost)
joblib.dump(pipeline, "/Users/lohithvattikuti/Desktop/Wine_Quality_project/outputs/stacked_model_no_xgboost.joblib")
print("Retrained model saved successfully!")