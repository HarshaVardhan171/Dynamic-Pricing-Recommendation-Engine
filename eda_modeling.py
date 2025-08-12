import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide", initial_sidebar_state="expanded")

# -------------------
# Data Loading & Caching
# -------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Dynamic Pricing Engine.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df['revenue'] = df['final_price'] * df['units_sold']
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

@st.cache_resource
def train_model(df):
    df_ml = pd.get_dummies(df.copy(), columns=['promotion'], drop_first=True)
    X = df_ml[['final_price', 'competitor_price', 'discount_percent', 'rating', 'promotion_True']]
    y = df_ml['units_sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {"R²": r2_score(y_test, y_pred), "MAE": mean_absolute_error(y_test, y_pred)}
    return model, metrics

model, metrics = train_model(df)

@st.cache_data
def get_clusters(df):
    feats = ['final_price', 'units_sold', 'discount_percent']
    X = StandardScaler().fit_transform(df[feats])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['price_cluster'] = kmeans.fit_predict(X)
    return df

df = get_clusters(df)

# -------------------
# App Layout
# -------------------
st.title("Dynamic Pricing Engine")
st.sidebar.success("Select a page above.")

tab1, tab2, tab3 = st.tabs(["💡 Prediction", "📊 Analytics", "📈 Elasticity Insights"])

# -------------------
# 1. Prediction Tab
# -------------------
with tab1:
    st.header("Predict Sales Volume")
    cols = st.columns(5)
    price = cols[0].slider("Set price:", 100, 5000, step=100)
    competitor_price = cols[1].slider("Competitor Price:", 100, 5000, step=100)
    discount = cols[2].slider("Discount %", 0, 50, step=5)
    rating = cols[3].slider("Rating", 1.0, 5.0, step=0.1)
    promo = cols[4].checkbox("Promotion Active?")
    input_df = pd.DataFrame([{
        'final_price': price,
        'competitor_price': competitor_price,
        'discount_percent': discount,
        'rating': rating,
        'promotion_True': int(promo)
    }])
    pred = model.predict(input_df)
    st.metric("Predicted Sales Volume", int(pred[0]))
    st.caption(f"Model R²: {metrics['R²']:.2f}, MAE: {metrics['MAE']:.0f}")

# -------------------
# 2. Analytics Tab
# -------------------
with tab2:
    st.header("Clustering Analysis")
    st.write("Products segmented by price, units sold, and discounts.")
    fig, ax = plt.subplots()
    sns.scatterplot(x='final_price', y='units_sold', hue='price_cluster', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)
    st.dataframe(df[['category', 'final_price', 'units_sold', 'discount_percent', 'price_cluster']])

# -------------------
# 3. Elasticity Insights Tab
# -------------------
with tab3:
    st.header("Price Elasticity by Category")
    eltab = df.groupby('category')[['final_price', 'units_sold']].mean()
    eltab['elasticity'] = (
        eltab['units_sold'].pct_change() / 
        eltab['final_price'].pct_change()
    )
    st.dataframe(eltab.style.format({"elasticity": "{:.2f}"}))
    st.markdown("""
        **Elasticity interpretation:**  
        - Elasticity < -1: highly price-sensitive  
        - Elasticity > -1: price-insensitive/luxury items
    """)

# -------------------
# End
# -------------------
