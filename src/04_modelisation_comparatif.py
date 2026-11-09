import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib


def load_numpy(path: str):
    return np.load(path, allow_pickle=True)


def plot_evaluation(name, cm, precision, recall, f1, out_dir):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    metrics = [precision, recall, f1]
    labels = ['Precision', 'Recall', 'F1-score']
    sns.barplot(x=labels, y=metrics, palette='pastel')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.title(f"{name} - Performance Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_metrics.png"))
    plt.close()


def evaluate_model(name, model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\nModel: {name}")
    print("Confusion matrix:")
    print(cm)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")
    plot_evaluation(name, cm, precision, recall, f1, out_dir)
    return {"model": name, "precision": precision, "recall": recall, "f1": f1}


def main():
    data_dir = 'data'
    print("===== ÉTAPE 4 : MODÉLISATION COMPARATIVE DES 4 ALGORITHMES =====")

    X_train = load_numpy(os.path.join(data_dir, 'X_train.npy'))
    X_test = load_numpy(os.path.join(data_dir, 'X_test.npy'))
    y_train = load_numpy(os.path.join(data_dir, 'y_train.npy'))
    y_test = load_numpy(os.path.join(data_dir, 'y_test.npy'))

    # Ensure 1-D arrays for targets
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    classifiers = [
        ("Logistic Regression", LogisticRegression(random_state=42, solver='liblinear')),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)),
        ("SVM", SVC(random_state=42))
    ]

    results = []
    graph_dir = 'graphs'
    os.makedirs(graph_dir, exist_ok=True)

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        res = evaluate_model(name, clf, X_test, y_test, graph_dir)
        results.append((res, clf))

    # Pick Random Forest as final model (per requirement)
    rf = None
    for res, clf in results:
        if res['model'] == 'Random Forest':
            rf = clf
            rf_metrics = res
            break

    if rf is None:
        print("Random Forest model not trained — aborting save.")
        return

    # Save model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'random_forest.joblib')
    joblib.dump(rf, model_path)
    print(f"\nSaved Random Forest model to {model_path}")
    print(f"Selected model metrics: Precision={rf_metrics['precision']:.3f}, Recall={rf_metrics['recall']:.3f}, F1={rf_metrics['f1']:.3f}")


if __name__ == '__main__':
    main()