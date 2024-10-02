# model_logistic_regression.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from scikeras.wrappers import KerasClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import joblib
import os

class xgb_model_classifier :

    # Function to load the dataset
    def load_data(self,file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"{file_path} not found!")

    # Preprocessing function
    def preprocess_historic_data(self,df):
        # Encode 'success_indicator' label (top = 1, flop = 0)
        label_encoder = LabelEncoder()
        df['success_indicator'] = label_encoder.fit_transform(df['success_indicator'])
        
        # One-Hot Encode the categorical columns ('category', 'main_promotion', 'color')
        df = pd.get_dummies(df, columns=['category', 'main_promotion', 'color'], 
                            prefix=['category', 'promotion', 'color'], drop_first=True)
        
        return df, label_encoder

    # Function to split data
    def split_data(self,df):
        X = df.drop(columns=['item_no', 'success_indicator'])
        y = df['success_indicator']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    # Model training function
    def train_model(self,X_train, y_train):
        xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgboost_model.fit(X_train, y_train)
        return xgboost_model

    # Evaluation function
    def evaluate_model(self,model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))

        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring='accuracy')
        print("Cross-Validation Accuracy Scores:", cv_scores)
        print("Mean CV Accuracy:", cv_scores.mean())
        

    # Function to save model and encoder
    def save_model(self,model, label_encoder, model_file="xgboost_model.pkl", encoder_file="label_encoder.pkl"):
        joblib.dump(model, model_file)
        joblib.dump(label_encoder, encoder_file)
        print(f"Model saved as {model_file} and label encoder as {encoder_file}")


if __name__ == "__main__":
    xgb_model = xgb_model_classifier()
    # Load the dataset
    file_path = "DSW_ML_Test/historic.csv"
    historic_df = xgb_model.load_data(file_path)

    # Preprocess data
    historic_df_processed, label_encoder = xgb_model.preprocess_historic_data(historic_df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = xgb_model.split_data(historic_df_processed)

    # Train the model
    xgb_class = xgb_model.train_model(X_train, y_train)

    # Evaluate the model
    xgb_model.evaluate_model(xgb_class, X_test, y_test)

    # Save the model and label encoder
    xgb_model.save_model(xgb_class, label_encoder)