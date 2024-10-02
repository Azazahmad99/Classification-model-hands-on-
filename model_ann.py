# model_ann.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier
import warnings
warnings.filterwarnings('ignore')

class ann_model_classifier:

    # Function to load the dataset
    def load_data(self, file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"{file_path} not found!")

    # Preprocessing function
    def preprocess_historic_data(self, df):
        # Encode 'success_indicator' label (top = 1, flop = 0)
        label_encoder = LabelEncoder()
        df['success_indicator'] = label_encoder.fit_transform(df['success_indicator'])
        
        # One-Hot Encode the categorical columns ('category', 'main_promotion', 'color')
        df = pd.get_dummies(df, columns=['category', 'main_promotion', 'color'], 
                            prefix=['category', 'promotion', 'color'], drop_first=True)

        # Standardize the features
        scaler = StandardScaler()
        X = df.drop(columns=['item_no', 'success_indicator'])
        X_scaled = scaler.fit_transform(X)
        df[X.columns] = X_scaled

        return df, label_encoder, scaler

    # Function to split data
    def split_data(self, df):
        X = df.drop(columns=['item_no', 'success_indicator'])
        y = df['success_indicator']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    # Function to create ANN model
    def create_ann_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Model training function
    def train_model(self, X_train, y_train):
        ann_model = self.create_ann_model(X_train.shape[1])
        history = ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
        return ann_model, history

    # Evaluation function
    def evaluate_model(self, model, X_test, y_test):
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))

        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_index, val_index in kf.split(X_test):
            X_val_train, X_val_test = X_test.iloc[train_index], X_test.iloc[val_index]
            y_val_train, y_val_test = y_test.iloc[train_index], y_test.iloc[val_index]
            val_loss, val_accuracy = model.evaluate(X_val_test, y_val_test, verbose=0)
            cv_scores.append(val_accuracy)
        print("Cross-Validation Accuracy Scores:", cv_scores)
        print("Mean CV Accuracy:", np.mean(cv_scores))

    # Function to save model, label encoder, and scaler
    def save_model(self, model, label_encoder, scaler, model_file="ann_model.h5", encoder_file="label_encoder.pkl", scaler_file="scaler.pkl"):
        model.save(model_file)
        joblib.dump(label_encoder, encoder_file)
        joblib.dump(scaler, scaler_file)
        print(f"Model saved as {model_file}, label encoder as {encoder_file}, and scaler as {scaler_file}")

if __name__ == "__main__":
    ann_model = ann_model_classifier()
    
    # Load the dataset
    file_path = "DSW_ML_Test/historic.csv"
    historic_df = ann_model.load_data(file_path)

    # Preprocess data
    historic_df_processed, label_encoder, scaler = ann_model.preprocess_historic_data(historic_df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = ann_model.split_data(historic_df_processed)

    # Train the model
    ann_classifier, history = ann_model.train_model(X_train, y_train)

    # Evaluate the model
    ann_model.evaluate_model(ann_classifier, X_test, y_test)

    # Save the model, label encoder, and scaler
    ann_model.save_model(ann_classifier, label_encoder, scaler)