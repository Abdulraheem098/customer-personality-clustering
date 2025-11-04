import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="Customer Segmentation App", layout="centered")
st.title("🧭 Customer Segmentation Using KMeans & PCA")

# -----------------------
# Load and Preprocess Dataset
# -----------------------
@st.cache_data
def load_and_preprocess():
    data = pd.read_excel("marketing_campaign.xlsx")

    # Drop missing values
    data = data.dropna()

    # Feature Engineering
    data['Education'] = data['Education'].replace(
        ['Graduation', 'PhD', 'Master', '2n Cycle'], 'Post Graduate')
    data['Education'] = data['Education'].replace(['Basic'], 'Under Graduate')

    data['Marital_Status'] = data['Marital_Status'].replace(
        ['Married', 'Together'], 'Relationship')
    data['Marital_Status'] = data['Marital_Status'].replace(
        ['Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')

    data['Kids'] = data['Kidhome'] + data['Teenhome']
    data['Expenses'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + \
                       data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
    data['TotalAcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + \
                               data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5']
    data['NumTotalPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + \
                                data['NumStorePurchases'] + data['NumDealsPurchases']
    data['customer_Age'] = pd.Timestamp('now').year - data['Year_Birth']

    # Drop unnecessary columns
    drop_cols = ['ID', 'Year_Birth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                 'AcceptedCmp4', 'AcceptedCmp5', 'NumWebVisitsMonth', 'NumWebPurchases',
                 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases',
                 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts',
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                 'Dt_Customer', 'Recency', 'Complain', 'Response',
                 'Z_CostContact', 'Z_Revenue']
    data = data.drop(columns=drop_cols, errors='ignore')

    # Columns
    cat_cols = ['Education', 'Marital_Status']
    num_cols = ['Income', 'Kids', 'Expenses',
                'TotalAcceptedCmp', 'NumTotalPurchases', 'customer_Age']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    # PCA + KMeans pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=3, random_state=42)),
        ('kmeans', KMeans(n_clusters=2, random_state=42))
    ])

    X = data[cat_cols + num_cols]
    pipeline.fit(X)

    return pipeline, cat_cols, num_cols


# Load model pipeline
pipeline, cat_cols, num_cols = load_and_preprocess()

# -----------------------
# Streamlit UI
# -----------------------
st.subheader("Enter New Customer Details")

# Input fields
Education = st.selectbox("Education Level", ["Under Graduate", "Post Graduate"])
Marital_Status = st.selectbox("Marital Status", ["Relationship", "Single"])
Income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
Kids = st.number_input("Number of Kids", min_value=0, max_value=10, value=0)
Expenses = st.number_input("Total Yearly Expenses", min_value=0.0, value=2000.0, step=100.0)
TotalAcceptedCmp = st.number_input("Total Accepted Campaigns", min_value=0, max_value=10, value=0)
NumTotalPurchases = st.number_input("Total Purchases", min_value=0, max_value=100, value=5)
customer_Age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)

# -----------------------
# Predict Cluster
# -----------------------
if st.button("🔍 Predict Customer Cluster"):
    # Create new customer DataFrame
    new_customer = pd.DataFrame([{
        "Education": Education,
        "Marital_Status": Marital_Status,
        "Income": Income,
        "Kids": Kids,
        "Expenses": Expenses,
        "TotalAcceptedCmp": TotalAcceptedCmp,
        "NumTotalPurchases": NumTotalPurchases,
        "customer_Age": customer_Age
    }])

    try:
        # Predict using full pipeline
        cluster = pipeline.predict(new_customer)[0]
        st.success(f"🎯 This customer belongs to **Cluster {cluster + 1}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------
# Display PCA clusters (optional)
# -----------------------
st.write("---")
st.subheader("📊 PCA-based Cluster Visualization (training data)")
st.write("Note: This shows how your training data was clustered using PCA & KMeans.")

if st.checkbox("Show Cluster Visualization"):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Extract original training data from pipeline
    pca_model = pipeline.named_steps['pca']
    kmeans_model = pipeline.named_steps['kmeans']
    preprocessor = pipeline.named_steps['preprocessor']

    # Transform training data for visualization
    data_vis = pd.read_excel("marketing_campaign.xlsx").dropna()
    data_vis['Education'] = data_vis['Education'].replace(
        ['Graduation', 'PhD', 'Master', '2n Cycle'], 'Post Graduate')
    data_vis['Education'] = data_vis['Education'].replace(['Basic'], 'Under Graduate')
    data_vis['Marital_Status'] = data_vis['Marital_Status'].replace(
        ['Married', 'Together'], 'Relationship')
    data_vis['Marital_Status'] = data_vis['Marital_Status'].replace(
        ['Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')

    data_vis['Kids'] = data_vis['Kidhome'] + data_vis['Teenhome']
    data_vis['Expenses'] = data_vis['MntWines'] + data_vis['MntFruits'] + data_vis['MntMeatProducts'] + \
                           data_vis['MntFishProducts'] + data_vis['MntSweetProducts'] + data_vis['MntGoldProds']
    data_vis['TotalAcceptedCmp'] = data_vis['AcceptedCmp1'] + data_vis['AcceptedCmp2'] + \
                                   data_vis['AcceptedCmp3'] + data_vis['AcceptedCmp4'] + data_vis['AcceptedCmp5']
    data_vis['NumTotalPurchases'] = data_vis['NumWebPurchases'] + data_vis['NumCatalogPurchases'] + \
                                    data_vis['NumStorePurchases'] + data_vis['NumDealsPurchases']
    data_vis['customer_Age'] = pd.Timestamp('now').year - data_vis['Year_Birth']

    X_vis = data_vis[cat_cols + num_cols]
    X_pca = pca_model.transform(preprocessor.transform(X_vis))
    labels = kmeans_model.predict(X_pca)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Clusters (PCA Projection)")
    plt.colorbar(scatter)
    st.pyplot(fig)
