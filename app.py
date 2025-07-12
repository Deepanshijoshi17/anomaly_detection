# anomaly_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Detection App", layout="wide")
st.title("🚨 Anomaly Detection using Isolation Forest")

# Upload CSV file
uploaded_file = st.file_uploader(r"C:\Users\joshi\Downloads\AnomalyDetection.ipynb", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(r"C:\Users\joshi\Downloads\AnomalyDetection.ipynb")

    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Allow user to select numeric features for detection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for anomaly detection.")
    else:
        selected_features = st.multiselect("Select numeric features to use", numeric_cols, default=numeric_cols)

        if selected_features:
            contamination = st.slider("Set contamination level (anomaly percentage)", 0.01, 0.5, 0.05, 0.01)

            X = df[selected_features]

            # Fit Isolation Forest
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            model.fit(X)

            df['anomaly'] = model.predict(X)
            df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

            st.success("✅ Anomaly detection completed!")

            # Show counts
            st.subheader("🔍 Anomaly Counts")
            st.write(df['anomaly'].value_counts())

            # Plotting
            if len(selected_features) == 2:
                st.subheader("📈 Scatter Plot of Anomalies")
                fig, ax = plt.subplots()
                sns.scatterplot(x=selected_features[0], y=selected_features[1], hue=df['anomaly'], palette=["red", "green"], ax=ax)
                st.pyplot(fig)
            else:
                st.info("Select exactly 2 features to see the scatter plot.")

            # Download option
            st.download_button("📥 Download Result CSV", df.to_csv(index=False), file_name="anomaly_detection_result.csv")

else:
    st.info(r"C:\Users\joshi\Downloads\AnomalyDetection.ipynb")