import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the trained model (XGBoost)
xgb_model = joblib.load('models/xgboost_model.joblib')

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Print classification report, confusion matrix, and accuracy score
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
