import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Login function
def check_login(username, password):
    if username == "admin" and password == "admin":
        return True
    else:
        return False

# Title of the dashboard
st.title('Centrifugal Pumps Data Analysis and Model Training')

# Display login form
st.subheader("Please login to access the dashboard")

# Collect username and password
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Button to submit login details
if st.button('Login'):
    if check_login(username, password):
        st.success("Login Successful")
        
        # Load the data from the local path
        file_path = r"C:\Users\Chkih\OneDrive\Desktop\PM2\Centrifugal_pumps_measurements.xlsx"
        df = pd.read_excel(file_path)

        # Show the first few rows of the dataframe
        st.subheader('First 5 rows of the dataset')
        st.write(df.head())

        # Show basic statistics
        st.subheader('Basic Statistics of Numerical Columns')
        st.write(df.describe())

        # Display Machine_ID count
        st.subheader('Machine_ID Distribution')
        machine_counts = df['Machine_ID'].value_counts()
        st.write(machine_counts)

        # Temperature statistics (valueTEMP)
        st.subheader('Temperature Statistics')
        st.write(df['valueTEMP'].describe())

        # Plotting the temperature distribution
        st.subheader('Temperature Distribution')
        plt.figure(figsize=(10, 6))
        sns.histplot(df['valueTEMP'], bins=20, kde=True)
        plt.title('Distribution of Temperature Values')
        plt.xlabel('Temperature')
        plt.ylabel('Frequency')
        st.pyplot()

        # Velocity statistics
        st.subheader('Velocity Statistics')
        st.write(df['velocity'].describe())

        # Plotting the velocity distribution
        st.subheader('Velocity Distribution')
        plt.figure(figsize=(10, 6))
        sns.histplot(df['velocity'], bins=20, kde=True)
        plt.title('Distribution of Velocity Values')
        plt.xlabel('Velocity')
        plt.ylabel('Frequency')
        st.pyplot()

        # Box Plot for all numerical columns
        st.subheader('Box Plot for All Numerical Columns')
        plt.figure(figsize=(12, 6))
        df.boxplot(rot=90)
        plt.title('Box Plot for All Variables')
        plt.ylabel('Values')
        st.pyplot()

        # Correlation heatmap
        st.subheader('Correlation Matrix Heatmap')
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        st.pyplot()

        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        st.subheader('Shuffled Data')
        st.write(df_shuffled.head())

        # Split data based on Machine_ID
        df_machine_1 = df[df['Machine_ID'] == 1]
        df_machine_2 = df[df['Machine_ID'] == 2]
        X_machine_1 = df_machine_1.drop(columns=['Machine_ID'])
        y_machine_2 = df_machine_2['Machine_ID'].map({1: 0, 2: 1})  # Map 1 to 0 (good), 2 to 1 (failure)

        # Train Test Split
        X_train, X_temp, y_train, y_temp = train_test_split(X_machine_1, y_machine_2, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

        # Apply Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest and XGBoost models
        st.subheader('Model Training')

        # Train Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_val_scaled)
        rf_accuracy = accuracy_score(y_val, rf_pred)
        st.write(f"Random Forest Accuracy: {rf_accuracy:.4f}")

        # Train XGBoost Classifier
        xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_val_scaled)
        xgb_accuracy = accuracy_score(y_val, xgb_pred)
        st.write(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    else:
        st.error("Invalid Username or Password")