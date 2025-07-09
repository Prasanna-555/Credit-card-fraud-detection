import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

print("🔄 Loading dataset...")
df = pd.read_csv("creditcard.csv").sample(n=10000, random_state=42)

print("📊 Preparing data...")
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

print(f"✅ Original shape: {X.shape}, {y.value_counts().to_dict()}")

print("⚖️ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("📈 Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("💾 Saving model...")
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")