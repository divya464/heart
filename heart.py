import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def load_model():
    with open("heart_attack_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_data(df):
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = df[features]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def main():
    st.title("Heart Attack Analysis & Prediction")
    st.write("Upload a CSV file to analyze heart attack risk factors and get predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.write(df.head())
        
        model = load_model()
        processed_data = preprocess_data(df)
        
        predictions = model.predict(processed_data)
        df['Heart Attack Risk'] = ['High' if pred == 1 else 'Low' for pred in predictions]
        
        st.write("### Predictions:")
        st.write(df[['age', 'sex', 'Heart Attack Risk']])
        
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
