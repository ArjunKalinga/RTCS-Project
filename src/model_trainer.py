import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration ---
DATA_FILE = os.path.join('data', 'classroom_actions.csv')
MODEL_FILE = os.path.join('models', 'attentiveness_model.joblib')
os.makedirs('models', exist_ok=True) # Create models directory if it doesn't exist

# --- 1. Load the Dataset ---
print(f"Loading dataset from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_FILE}")
    print("Please run the dataset_processor.py script first.")
    exit()

# --- FIX: Instead of dropping rows, fill missing values with 0 ---
df.fillna(0, inplace=True)

if df.empty:
    print("Error: The dataset is still empty after loading. Please check the CSV file.")
    exit()

print(f"Dataset loaded successfully with {len(df)} samples.")
print("\nClass distribution:")
print(df['label'].value_counts())

# --- 2. Prepare the Data ---
# Separate features (keypoints) from the label
X = df.drop('label', axis=1) # All columns except 'label' are features
y = df['label']              # The 'label' column is our target

# --- 3. Split Data into Training and Testing Sets ---
# 80% for training, 20% for testing. random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- 4. Train the Model ---
# We use a RandomForestClassifier, which is a powerful and reliable model for this kind of task.
print("\nTraining the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Trained Model ---
print(f"\nSaving the trained model to {MODEL_FILE}...")
joblib.dump(model, MODEL_FILE)
print("Model saved successfully.")

