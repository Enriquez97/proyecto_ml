import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_churn_model(data_path, max_iter=1000, c_parameter=1.0):
    """
    Entrena un modelo de Regresión Logística para predecir Churn.
    """
    
    # 1. Cargar Datos
    print("Cargando datos...")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        return None, f"Error al cargar datos: {e}"

    
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(0)

    
    target = 'Churn Value'
    numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 
                            'Phone Service', 'Multiple Lines', 'Internet Service', 
                            'Online Security', 'Online Backup', 'Device Protection', 
                            'Tech Support', 'Streaming TV', 'Streaming Movies', 
                            'Contract', 'Paperless Billing', 'Payment Method']

   
    missing_cols = [col for col in numeric_features + categorical_features + [target] if col not in df.columns]
    if missing_cols:
        return None, f"Faltan columnas en el dataset: {missing_cols}"
        
    X = df[numeric_features + categorical_features]
    y = df[target]

    
    print("Creando pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(random_state=42, C=c_parameter, max_iter=max_iter))])

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    print("Entrenando modelo...")
    clf.fit(X_train, y_train)

    
    print("Evaluando modelo...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    
    print("Guardando modelo...")
    joblib.dump(clf, 'churn_model_pipeline.pkl')
    print("Modelo guardado en churn_model_pipeline.pkl")
    
    return clf, {
        "accuracy": accuracy,
        "report": report
    }

if __name__ == "__main__":
    
    model, metrics = train_churn_model(r'./data/Telco_customer_churn.xlsx')
    if model:
        print(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.4f}")
    else:
        print(metrics) 
