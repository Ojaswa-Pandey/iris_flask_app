import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib 

# Load data
df = pd.read_csv("iris.data", names=["sepal length", "sepal width", "petal length", "petal width", "class"])
X = df.iloc[:,0:4]
y = df.iloc[:,-1]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Save model and label encoder
joblib.dump(model, "iris_model.joblib")
joblib.dump(encoder, "label_encoder.joblib")

print("Model and encoder saved successfully.")
