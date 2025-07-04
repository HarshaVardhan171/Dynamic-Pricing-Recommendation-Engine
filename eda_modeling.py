import sklearn; print(sklearn.__version__)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load Data
def load_data():
    df = pd.read_csv("Dynamic Pricing Recom Engine.csv")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df['revenue'] = df['final_price'] * df['units_sold']
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

# Select features (remove 'promotion' because it will be replaced by 'promotion_True')
features = ['final_price', 'competitor_price', 'discount_percent', 'rating']
df_ml = df.copy()
df_ml = pd.get_dummies(df_ml, columns=['promotion'], drop_first=True)

X = df_ml[features + ['promotion_True']]
y = df_ml['units_sold']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Group by product or category
df_elasticity = df.groupby('category')[['final_price', 'units_sold']].mean()
df_elasticity['elasticity'] = (
    df_elasticity['units_sold'].pct_change() / df_elasticity['final_price'].pct_change()
)
df_elasticity

#Elasticity Interpretation
    #Elasticity < -1 → highly price-sensitive
    #Elasticity > -1 → price-insensitive or luxury items

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_cluster = df[['final_price', 'units_sold', 'discount_percent']]
X_scaled = StandardScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3)
df['price_cluster'] = kmeans.fit_predict(X_scaled)

from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)

price = st.slider("Set price:", 100, 5000, step=100)
competitor_price = st.slider("Competitor Price:", 100, 5000, step=100)
discount = st.slider("Discount %", 0, 50, step=5)
rating = st.slider("Rating", 1.0, 5.0, step=0.1)
promo = st.checkbox("Promotion Active?")

input_dict = {
    'final_price': price,
    'competitor_price': competitor_price,
    'discount_percent': discount,
    'rating': rating,
    'promotion_True': int(promo)
}
input_df = pd.DataFrame([input_dict])

predicted_sales = model.predict(input_df)
st.metric("Predicted Sales Volume", int(predicted_sales[0]))
