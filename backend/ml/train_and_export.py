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
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_alzheimer_model():
    print("=" * 60)
    print("Training Alzheimer\'s Disease Prediction Model...")
    csv_path = os.path.join(DATA_DIR, "alzheimers_disease_data.csv")
    df = pd.read_csv(csv_path)

    drop_cols = [
        'PatientID', 'DoctorInCharge', 'CholesterolLDL', 'Ethnicity', 
        'CholesterolHDL', 'CholesterolTriglycerides', 'Forgetfulness'
    ]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    X = df_clean.drop(columns=['Diagnosis'])
    y = df_clean['Diagnosis']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=35, shuffle=True
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=0, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_prob)

    print(f"Alzheimer Model Performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    coefs = pipe.named_steps['clf'].coef_[0]
    importance = {feat: float(coef) for feat, coef in zip(feature_names, coefs)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    artifact = {
        "pipeline": pipe,
        "features": feature_names,
        "target": "Diagnosis",
        "classes": [0, 1],
        "class_labels": {0: "Low Risk / No Alzheimer's", 1: "High Risk / Alzheimer's Indicated"},
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

    out_path = os.path.join(MODELS_DIR, "alzheimer_model.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved Alzheimer model to: {out_path}")
    return artifact


def train_diabetes_model():
    print("=" * 60)
    print("Training Diabetes Prediction Model...")
    csv_path = os.path.join(DATA_DIR, "diabetes.csv")
    df = pd.read_csv(csv_path)

    drop_cols = ["Outcome", "Insulin", "BMI", "DiabetesPedigreeFunction"]
    X = df.drop(columns=drop_cols)
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_prob)

    print(f"Diabetes Model Performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    coefs = pipe.named_steps['clf'].coef_[0]
    importance = {feat: float(coef) for feat, coef in zip(feature_names, coefs)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    artifact = {
        "pipeline": pipe,
        "features": feature_names,
        "target": "Outcome",
        "classes": [0, 1],
        "class_labels": {0: "Non-Diabetic / Low Risk", 1: "Diabetic / Elevated Risk"},
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

    out_path = os.path.join(MODELS_DIR, "diabetes_model.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved Diabetes model to: {out_path}")
    return artifact


def train_cardio_model():
    print("=" * 60)
    print("Training Cardiovascular Disease Model...")
    csv_path = os.path.join(DATA_DIR, "heart_disease_data.csv")
    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        low = c.lower()
        if low == 'datasource': col_map[c] = 'DataSource'
        elif low == 'confidence_limit_low': col_map[c] = 'LowConfidenceLimit'
        elif low == 'confidence_limit_high': col_map[c] = 'HighConfidenceLimit'
        elif low == 'break_out': col_map[c] = 'Break_Out'
        elif low == 'break_out_category': col_map[c] = 'Break_Out_Category'
        elif low == 'data_value_unit': col_map[c] = 'Data_Value_Unit'
        elif low == 'data_value_alt': col_map[c] = 'Data_Value_Alt'
    if col_map:
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
        if c in df_sub.columns:
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
    X = df_sub[feature_cols]
    y = df_sub['Topic_Encoded']

    if len(df_sub) > 10000:
        df_sample = df_sub.sample(n=10000, random_state=100)
        X = df_sample[feature_cols]
        y = df_sample['Topic_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)
    acc = metrics.accuracy_score(y_test, y_pred)

    print(f"Cardiovascular KNN+PCA Model Accuracy: {acc:.4f}")
    target_classes = {int(i): str(cls_name) for i, cls_name in enumerate(target_le.classes_)}

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
    print(f"Saved Cardiovascular pipeline to: {out_path}")
    return artifact


if __name__ == "__main__":
    train_alzheimer_model()
    train_diabetes_model()
    train_cardio_model()
    print("All 3 models trained and exported successfully!")
