import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def connect_to_db():
    # Using your Railway connection string here
    engine = create_engine('postgresql://postgres:spTFufHenHnSBDaNHIZnhmiWPpGonlKW@tramway.proxy.rlwy.net:45731/railway')
    return engine.connect()

@st.cache_data
def load_data():
    try:
        connection = connect_to_db()
        query = "SELECT * FROM final_fooddata"
        df = pd.read_sql(query, connection)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data loaded. Please check your database connection.")
    st.stop()

customer_ids = df['Customer ID'].unique()
selected_customer = st.sidebar.selectbox("Select Customer ID", customer_ids)

# -------------------------------
# Fast and Slow-Moving Items
# -------------------------------
st.header("📦 Fast and Slow-Moving Items")
product_sales = df.groupby("Product")["Quantity"].sum().reset_index()
threshold = product_sales["Quantity"].median()
product_sales["Category"] = product_sales["Quantity"].apply(
    lambda x: "Fast-Moving" if x >= threshold else "Slow-Moving"
)

show_fast = st.button("Show Fast-Moving Items")
show_slow = st.button("Show Slow-Moving Items")

fig1, ax1 = plt.subplots(figsize=(14, 6))

if show_fast:
    fast_df = product_sales[product_sales["Category"] == "Fast-Moving"].sort_values("Quantity", ascending=False)
    st.write("### Fast-Moving Items")
    st.dataframe(fast_df)
    sns.barplot(data=fast_df.head(30), x="Product", y="Quantity", color="orange", ax=ax1)
    ax1.set_title("Top 30 Fast-Moving Products")
elif show_slow:
    slow_df = product_sales[product_sales["Category"] == "Slow-Moving"].sort_values("Quantity", ascending=False)
    st.write("### Slow-Moving Items")
    st.dataframe(slow_df)
    sns.barplot(data=slow_df.head(30), x="Product", y="Quantity", color="gray", ax=ax1)
    ax1.set_title("Top 30 Slow-Moving Products")
else:
    combined = product_sales.sort_values("Quantity", ascending=False).head(30)
    sns.barplot(data=combined, x="Product", y="Quantity", hue="Category", ax=ax1)
    ax1.set_title("Top 30 Products - Mixed")

plt.xticks(rotation=90)
st.pyplot(fig1)

# -------------------------------
# Seasonal Sales Analysis
# -------------------------------
st.header("📊 Seasonal Sales Analysis of Products")
df['Month'] = df['Date'].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby(["Product", "Month"])["Quantity"].sum().reset_index()
products = df["Product"].unique()
selected_product = st.selectbox("Select Product for Seasonal Analysis", products)

product_df = monthly[monthly["Product"] == selected_product]
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(data=product_df, x="Month", y="Quantity", palette="coolwarm", ax=ax2)
ax2.set_title(f"Monthly Sales Trend: {selected_product}")
plt.xticks(rotation=45)
st.pyplot(fig2)

# -------------------------------
# 📅 Customer Next Purchase Section
# -------------------------------
st.header("📅 Customer Next Purchase")

next_purchase_records = []
for cust_id in df["Customer ID"].unique():
    customer_df = df[df["Customer ID"] == cust_id]
    for prod in customer_df["Product"].unique():
        prod_df = customer_df[customer_df["Product"] == prod]
        purchase_dates = prod_df["Date"].sort_values()
        if len(purchase_dates) > 1:
            gaps = purchase_dates.diff().dropna().dt.days
            avg_gap = gaps.mean()
            next_purchase_date = purchase_dates.max() + pd.Timedelta(days=avg_gap)
            next_purchase_date = next_purchase_date.date()
        else:
            next_purchase_date = "Not enough data"

        next_purchase_records.append({
            "Customer ID": cust_id,
            "Product": prod,
            "Predicted Next Purchase Date": next_purchase_date
        })

next_purchase_df = pd.DataFrame(next_purchase_records)
st.dataframe(next_purchase_df)

# -------------------------------
# 🔮 Customer Stock Prediction
# -------------------------------
st.header("🔮 Customer Stock Prediction for All Products")
st.subheader(f"Selected Customer ID: {selected_customer}")

customer_df = df[df["Customer ID"] == selected_customer]
products_for_customer = customer_df["Product"].unique()

for prod in products_for_customer:
    prod_df = customer_df[customer_df["Product"] == prod]
    time_series = prod_df.resample("M", on="Date")["Quantity"].sum()

    if len(time_series.dropna()) >= 3:
        try:
            model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=1)[0]
            st.metric(label=f"{prod}", value=f"{forecast:.2f}")
        except:
            avg_qty = time_series.mean()
            st.metric(label=f"{prod} (Avg)", value=f"{avg_qty:.2f}")
    else:
        fallback = prod_df.groupby(prod_df["Date"].dt.to_period("M"))["Quantity"].sum().mean()
        st.metric(label=f"{prod} (Est Avg)", value=f"{fallback:.2f}")
