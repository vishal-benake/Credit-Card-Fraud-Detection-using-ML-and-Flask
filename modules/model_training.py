import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib



def train_model(df):
    x = df.drop('Class', axis=1)
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

     # Pipeline with scaler + logistic regression
    pipeline = ImbPipeline([
         ('smote', SMOTE()),
         ('scaler', StandardScaler()),
         ('clf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1))
     ])

    pipeline.fit(x_train, y_train)

     # Save test set for evaluation
    return pipeline, x_test, y_test

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
