# model_logistic_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import joblib
import os

class logistic_regression_model :

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
        logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        logreg_model.fit(X_train, y_train)
        return logreg_model

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
    def save_model(self,model, label_encoder, model_file="logistic_regression_model.pkl", encoder_file="label_encoder.pkl"):
        joblib.dump(model, model_file)
        joblib.dump(label_encoder, encoder_file)
        print(f"Model saved as {model_file} and label encoder as {encoder_file}")


if __name__ == "__main__":
    lrm = logistic_regression_model()
    # Load the dataset
    file_path = "DSW_ML_Test/historic.csv"
    historic_df = lrm.load_data(file_path)

    # Preprocess data
    historic_df_processed, label_encoder = lrm.preprocess_historic_data(historic_df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = lrm.split_data(historic_df_processed)

    # Train the model
    logreg_model = lrm.train_model(X_train, y_train)

    # Evaluate the model
    lrm.evaluate_model(logreg_model, X_test, y_test)

    # Save the model and label encoder
    lrm.save_model(logreg_model, label_encoder)