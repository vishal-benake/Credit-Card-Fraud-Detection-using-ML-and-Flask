from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))