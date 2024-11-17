import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Data scaling
scaler = StandardScaler()
df_scaled = df.copy()
feature_columns = [col for col in df.columns if col != 'Class']
df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

# Split into features (X) and target (y)
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))

# Save the best model (let's save XGBoost model in this example)
joblib.dump(xgb_model, 'models/xgboost_model.joblib')
