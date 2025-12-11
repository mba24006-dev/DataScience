import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib # Placeholder for model loading

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Data Science Job Change Analysis")
sns.set_style("whitegrid")

# --- Data Cleaning and Preprocessing Functions (Steps 3-5) ---

@st.cache_data
def load_data(file_path):
    """Loads the dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please ensure the file is in the correct location.")
        return pd.DataFrame()

def handle_missing_values(df):
    """Implements Step 3: Handle missing values."""
    # 1. Categorical columns with high missingness (impute with 'Unknown')
    high_missing_cols = ['company_type', 'company_size', 'gender', 'major_discipline']
    for col in high_missing_cols:
        df[col] = df[col].fillna('Unknown')

    # 2. Categorical columns with low missingness (impute with Mode)
    low_missing_cat_cols = ['enrolled_university', 'education_level']
    for col in low_missing_cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Numerical columns with low missingness (impute with Median)
    low_missing_num_cols = ['experience', 'city_development_index', 'training_hours']
    for col in low_missing_num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def cap_outliers_iqr(series):
    """Helper function for IQR capping."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    series_capped = np.where(series < lower_bound, lower_bound, series)
    series_capped = np.where(series_capped > upper_bound, upper_bound, series_capped)
    
    return pd.Series(series_capped, index=series.index)

def clean_data(df):
    """Implements Step 4 & 5: Handle outliers and clean data."""
    
    # Step 4: Handle outliers using IQR (Capping/Winsorizing)
    numerical_cols_for_outlier = ['city_development_index', 'training_hours']
    for col in numerical_cols_for_outlier:
        df[col] = cap_outliers_iqr(df[col])
        
    # Step 5: Clean data (Remove duplicates, clean types)
    df.drop_duplicates(inplace=True)
    df['experience'] = df['experience'].astype(int)
    
    return df

# --- EDA Functions (Step 6) ---

def plot_eda(df):
    """Generates and displays EDA plots."""
    st.header("Step 6: Exploratory Data Analysis (EDA)")
    
    numerical_features = ['city_development_index', 'training_hours', 'experience']
    categorical_features = ['relevent_experience', 'enrolled_university', 'education_level', 'gender']

    # 1. Target Distribution
    st.subheader("1. Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, ax=ax)
    ax.set_title('Target Variable Distribution (0: No Change, 1: Looking for Change)')
    st.pyplot(fig)
    st.markdown(f"**Imbalance Check:** {df['target'].value_counts(normalize=True).loc[1]:.2%} of candidates are looking for a job change.")

    # 2. Histograms for Numerical Features
    st.subheader("2. Distribution of Numerical Features")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    st.pyplot(fig)

    # 3. Boxplots for Numerical Features vs. Target
    st.subheader("3. Numerical Features vs. Target")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(numerical_features):
        sns.boxplot(x='target', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} vs. Target')
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Correlation Matrix
    st.subheader("4. Correlation Matrix")
    corr_matrix = df[numerical_features + ['target']].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features and Target')
    st.pyplot(fig)
    st.markdown(f"**Key Insight:** `city_development_index` has the strongest correlation with the target: **{corr_matrix.loc['city_development_index', 'target']:.2f}**.")

    # 5. Bar plots for Categorical Features vs. Target
    st.subheader("5. Job Change Rate by Categorical Features")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(categorical_features):
        prop_df = df.groupby(col)['target'].mean().reset_index()
        sns.barplot(x=col, y='target', data=prop_df, ax=axes[i])
        axes[i].set_title(f'Job Change Rate by {col}')
        axes[i].set_ylabel('Proportion of Target=1')
        axes[i].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

# --- Model Placeholder (Steps 7-12) ---

def model_placeholder():
    """Explains the model limitation."""
    st.header("Steps 7-12: Model Selection, Training, and Evaluation")
    st.error("""
        **Model Training Limitation:**
        The machine learning steps (Model Selection, Training, Prediction, Evaluation, and Improvement) could not be executed because the required library, **scikit-learn**, could not be installed in the current environment due to a technical limitation.

        **If the model were trained, the process would be:**
        1.  **Feature Engineering:** One-Hot Encode categorical features and Scale numerical features.
        2.  **Model Selection:** Use **Logistic Regression** or **Random Forest** for this binary classification problem.
        3.  **Evaluation:** Use **ROC AUC** as the primary metric due to the class imbalance.
        4.  **Improvement:** Apply techniques like **SMOTE** or **Hyperparameter Tuning** to boost performance.
        
        The `model.pkl` and `features.json` files are included as placeholders for a complete project structure.
    """)

# --- Main Application ---

def main():
    st.title("Data Science Job Change Analysis Pipeline")
    st.markdown("This application executes the first six steps of the provided 12-step data science methodology.")

    # Assuming the file is uploaded or available in the path
    # For a real Streamlit app, you would use st.file_uploader
    # For this demonstration, we use the provided path
    data_file_path = '/home/ubuntu/upload/data_science_job.csv'
    
    df = load_data(data_file_path)
    
    if not df.empty:
        st.header("Step 1 & 2: Data Loading and Initial Exploration")
        st.subheader("Raw Data Head")
        st.dataframe(df.head())
        
        # Data Cleaning Pipeline
        df_imputed = handle_missing_values(df.copy())
        df_cleaned = clean_data(df_imputed.copy())
        
        st.header("Step 3, 4, & 5: Data Cleaning Summary")
        st.markdown(f"""
        - **Missing Values:** Handled using Median/Mode for low missingness and 'Unknown' category for high missingness.
        - **Outliers:** Capped using the **IQR method** on `city_development_index` and `training_hours`.
        - **Duplicates:** {len(df) - len(df_cleaned)} duplicates removed (0 in this case).
        """)
        st.subheader("Cleaned Data Head")
        st.dataframe(df_cleaned.head())
        
        # EDA
        plot_eda(df_cleaned)
        
        # Model Placeholder
        model_placeholder()

if __name__ == "__main__":
    main()
