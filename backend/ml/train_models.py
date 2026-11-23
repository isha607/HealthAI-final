import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================================================================
# 1. ALZHEIMER'S DISEASE MODEL (alzheimer-s-disease-prediction (1).ipynb)
# ==============================================================================
def train_and_export_alzheimer():
    print("=" * 60)
    print("Training Alzheimer's Disease Prediction Model from notebook logic...")
    csv_path = os.path.join(DATA_DIR, "alzheimers_disease_data.csv")
    df = pd.read_csv(csv_path)

    # In notebook: df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
    df_clean = df.drop(columns=['PatientID', 'DoctorInCharge'])
    X = df_clean.drop(columns=['Diagnosis'])
    y = df_clean['Diagnosis']
    feature_names = X.columns.tolist()

    continuous_columns = [
        'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
        'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
        'MMSE', 'FunctionalAssessment', 'ADL'
    ]

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[continuous_columns] = scaler.fit_transform(X[continuous_columns])

    # In notebook: test_size = 0.2, random_state = 42, shuffle = True
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Notebook best parameters for Logistic Regression: C=1.0, random_state=0
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_prob)

    print(f"Alzheimer Model Performance:")
    print(f"  Accuracy : {acc:.4f} (Notebook reported: 0.83)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f} (Notebook reported: 0.81)")
    print(f"  ROC-AUC  : {auc:.4f}")

    coefs = clf.coef_[0]
    importance = {feat: float(coef) for feat, coef in zip(feature_names, coefs)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    artifact = {
        "model": clf,
        "scaler": scaler,
        "continuous_columns": continuous_columns,
        "features": feature_names,
        "target": "Diagnosis",
        "classes": [0, 1],
        "class_labels": {0: "Low Risk / No Alzheimer's Indicated", 1: "Increased Statistical Risk of Alzheimer's"},
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc)
        },
        "feature_importance": sorted_importance,
        "version": "1.0.0"
    }

    out_path = os.path.join(MODELS_DIR, "alzheimer_pipeline.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved Alzheimer artifact to: {out_path}")
    return artifact


# ==============================================================================
# 2. DIABETES PREDICTION MODEL (diabetes-prediction-logistic-regression.ipynb)
# ==============================================================================
def train_and_export_diabetes():
    print("=" * 60)
    print("Training Diabetes Prediction Model from notebook logic...")
    csv_path = os.path.join(DATA_DIR, "diabetes.csv")
    df = pd.read_csv(csv_path)

    # In notebook: X=df.drop(columns=["Outcome","Insulin","BMI","DiabetesPedigreeFunction"])
    drop_cols = ["Outcome", "Insulin", "BMI", "DiabetesPedigreeFunction"]
    X = df.drop(columns=drop_cols)
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    # In notebook: scaler=StandardScaler(); X=scaler.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # In notebook: X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.12)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.12, random_state=42
    )

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_prob)

    print(f"Diabetes Model Performance:")
    print(f"  Accuracy : {acc:.4f} (Notebook reported: ~0.80)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    coefs = clf.coef_[0]
    importance = {feat: float(coef) for feat, coef in zip(feature_names, coefs)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    artifact = {
        "model": clf,
        "scaler": scaler,
        "features": feature_names,
        "target": "Outcome",
        "classes": [0, 1],
        "class_labels": {0: "Low Risk / Non-Diabetic", 1: "Increased Statistical Risk of Diabetes"},
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc)
        },
        "feature_importance": sorted_importance,
        "version": "1.0.0"
    }

    out_path = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved Diabetes artifact to: {out_path}")
    return artifact


# ==============================================================================
# 3. CARDIOVASCULAR DISEASE MODEL (cardiovascular-disease-prediction.ipynb)
# ==============================================================================
def train_and_export_cardio():
    print("=" * 60)
    print("Training Cardiovascular Disease Model from notebook logic...")
    csv_path = os.path.join(DATA_DIR, "heart_disease_data.csv")
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names to match the exact notebook variables
    col_map = {
        'Break_out': 'Break_Out',
        'Confidence_Limit_Low': 'LowConfidenceLimit',
        'Confidence_Limit_High': 'HighConfidenceLimit',
        'Datasource': 'DataSource'
    }
    df.rename(columns=col_map, inplace=True)

    drop_cols = [
        'Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'GeoLocation', 'Data_Value',
        'PriorityArea2', 'PriorityArea4', 'DataSource', 'Category', 'Data_Value_Type',
        'LocationAbbr', 'LocationDesc', 'TopicId', 'TopicID', 'BreakOutId', 'BreakOutID',
        'BreakOutCategoryId', 'BreakoutCategoryID', 'CategoryId', 'CategoryID',
        'LocationID', 'Data_Value_TypeID', 'IndicatorID', 'Million Hearts',
        'Healthy People 2020', 'PriorityArea1', 'PriorityArea3'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    if 'PriorityArea' not in df.columns:
        df['PriorityArea'] = 'General'

    req_cols = [
        'Year', 'Topic', 'Indicator', 'Data_Value_Unit', 'Data_Value_Alt',
        'LowConfidenceLimit', 'HighConfidenceLimit', 'Break_Out_Category',
        'Break_Out', 'PriorityArea'
    ]
    df_sub = df[[c for c in req_cols if c in df.columns]].dropna().copy()

    cat_cols = ['Indicator', 'Data_Value_Unit', 'Break_Out_Category', 'Break_Out', 'PriorityArea']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_sub[c] = le.fit_transform(df_sub[c].astype(str))
        encoders[c] = le

    target_le = LabelEncoder()
    df_sub['Topic_Encoded'] = target_le.fit_transform(df_sub['Topic'].astype(str))
    encoders['Topic'] = target_le

    feature_cols = [
        'Year', 'Indicator', 'Data_Value_Unit', 'Data_Value_Alt',
        'LowConfidenceLimit', 'HighConfidenceLimit', 'Break_Out_Category',
        'Break_Out', 'PriorityArea'
    ]

    # Sample for robust and fast reproducibility
    if len(df_sub) > 10000:
        df_sample = df_sub.sample(n=10000, random_state=100)
        X = df_sample[feature_cols]
        y = df_sample['Topic_Encoded']
    else:
        X = df_sub[feature_cols]
        y = df_sub['Topic_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # In notebook: knn = KNeighborsClassifier(n_neighbors=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)
    acc = metrics.accuracy_score(y_test, y_pred)

    print(f"Cardiovascular KNN+PCA Model Accuracy: {acc:.4f}")
    target_classes = {int(i): str(cls_name) for i, cls_name in enumerate(target_le.classes_)}
    print(f"Classes: {target_classes}")

    artifact = {
        "scaler": scaler,
        "pca": pca,
        "knn": knn,
        "encoders": encoders,
        "features": feature_cols,
        "target": "Topic",
        "classes": target_classes,
        "metrics": {"accuracy": float(acc)},
        "version": "1.0.0"
    }

    out_path = os.path.join(MODELS_DIR, "cardio_pipeline.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved Cardiovascular artifact to: {out_path}")
    return artifact


if __name__ == "__main__":
    train_and_export_alzheimer()
    train_and_export_diabetes()
    train_and_export_cardio()
    print("All 3 models trained and exported successfully!")
