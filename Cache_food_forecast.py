import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
@st.cache_resource
def connect_to_db():
    engine = create_engine('postgresql://postgres:spTFufHenHnSBDaNHIZnhmiWPpGonlKW@tramway.proxy.rlwy.net:45731/railway')
    return engine.connect()

@st.cache_data(ttl=600)
def load_all_data():
    conn = connect_to_db()
    df = pd.read_sql("SELECT * FROM final_fooddata", conn)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_all_data()
if df.empty:
    st.error("No data found.")
    st.stop()

customer_ids = df["Customer ID"].unique()
selected_customer = st.sidebar.selectbox("Select Customer ID", customer_ids)
max_customers = st.sidebar.slider("Max Customers for SARIMA", 1, 50, 10)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Dashboard (All Customers)", "Customer Analysis"])

# -------------------------------
# SARIMA CACHE
# -------------------------------
@st.cache_data(ttl=1800)
def sarima_forecast(ts):
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    result = model.fit(disp=False)
    return result.forecast(steps=1)[0]

# -------------------------------
# TAB 1: Full Dashboard
# -------------------------------
with tab1:
    st.header("📦 Fast and Slow-Moving Items (All Customers)")
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
        st.dataframe(fast_df)
        sns.barplot(data=fast_df.head(30), x="Product", y="Quantity", color="orange", ax=ax1)
        ax1.set_title("Top 30 Fast-Moving Products")
    elif show_slow:
        slow_df = product_sales[product_sales["Category"] == "Slow-Moving"].sort_values("Quantity", ascending=False)
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
    st.header("📊 Seasonal Sales Trends (All Customers)")
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["Product", "Month"])["Quantity"].sum().reset_index()
    selected_product = st.selectbox("Select Product for Seasonal Analysis", df["Product"].unique())
    seasonal_df = monthly[monthly["Product"] == selected_product]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=seasonal_df, x="Month", y="Quantity", palette="coolwarm", ax=ax2)
    ax2.set_title(f"Monthly Sales Trend: {selected_product}")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # -------------------------------
    st.header("📅 Next Purchase Prediction (All Customers)")
    if st.button("Run Next Purchase Prediction"):
        next_purchase_records = []
        for cust_id in df["Customer ID"].unique():
            cust_df = df[df["Customer ID"] == cust_id]
            for prod in cust_df["Product"].unique():
                prod_df = cust_df[cust_df["Product"] == prod]
                purchase_dates = prod_df["Date"].sort_values()
                if len(purchase_dates) > 1:
                    gaps = purchase_dates.diff().dropna().dt.days
                    avg_gap = gaps.mean()
                    next_date = purchase_dates.max() + pd.Timedelta(days=avg_gap)
                    next_date = next_date.date()
                else:
                    next_date = "Not enough data"
                next_purchase_records.append({
                    "Customer ID": cust_id,
                    "Product": prod,
                    "Next Purchase": next_date
                })
        st.dataframe(pd.DataFrame(next_purchase_records))

    # -------------------------------
    st.header("🔮 Stock Forecast for All Customers")
    for cust_id in customer_ids[:max_customers]:
        st.subheader(f"Customer ID: {cust_id}")
        cust_df = df[df["Customer ID"] == cust_id]
        for prod in cust_df["Product"].unique():
            prod_df = cust_df[cust_df["Product"] == prod]
            ts = prod_df.resample("M", on="Date")["Quantity"].sum()
            if len(ts.dropna()) >= 4:
                try:
                    forecast = sarima_forecast(ts)
                    st.metric(label=f"{prod}", value=f"{forecast:.2f}")
                except:
                    fallback = ts.mean()
                    st.metric(label=f"{prod} (Avg)", value=f"{fallback:.2f}")
            else:
                fallback = ts.mean()
                st.metric(label=f"{prod} (Est Avg)", value=f"{fallback:.2f}")

# -------------------------------
# TAB 2: Per-Customer Analysis
# -------------------------------
with tab2:
    st.header(f"📈 Purchase Pattern for Customer: {selected_customer}")
    customer_data = df[df["Customer ID"] == selected_customer]
    if customer_data.empty:
        st.write("No data available.")
    else:
        st.subheader("Purchase Frequency per Product")
        freq_df = customer_data.groupby("Product")["Date"].count().reset_index(name="Purchase Count")
        st.dataframe(freq_df)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=freq_df, x="Product", y="Purchase Count", palette="viridis", ax=ax3)
        plt.xticks(rotation=90)
        st.pyplot(fig3)

        st.subheader("Avg Purchase Interval (Days) per Product")
        intervals = []
        for prod in customer_data["Product"].unique():
            prod_dates = customer_data[customer_data["Product"] == prod]["Date"].sort_values()
            avg_interval = prod_dates.diff().dt.days.mean() if len(prod_dates) > 1 else None
            intervals.append({"Product": prod, "Avg Interval (days)": avg_interval})
        st.dataframe(pd.DataFrame(intervals))

        st.subheader("Monthly Purchase Heatmap")
        customer_data["Month"] = customer_data["Date"].dt.to_period("M").dt.to_timestamp()
        heatmap_data = customer_data.pivot_table(index="Product", columns="Month", values="Quantity", aggfunc="sum", fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax4)
        st.pyplot(fig4)
