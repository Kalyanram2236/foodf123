import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")
# DATABASE CONNECTION
@st.cache_resource
def connect_to_db():
    engine = create_engine('postgresql://postgres:FsJwKKUUDSWHADQyCgnsjYjPneYknyyx@nozomi.proxy.rlwy.net:13925/railway')
    return engine.connect()

@st.cache_data(ttl=600)
def load_all_data():
    conn = connect_to_db()
    df = pd.read_sql("SELECT * FROM ramp", conn)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])

@st.cache_data(ttl=1000)
def preprocess_monthly_data(df):
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["Product", "Month"])["Quantity"].sum().reset_index()

@st.cache_data(ttl=1000)
def assign_seasonal_tags(df):
    df = df.copy()
    df["Weather_Season"] = df["Date"].dt.month.apply(assign_weather_season)
    df["Festival_Season"] = df["Date"].apply(assign_festival_season)
    return df

@st.cache_data(ttl=2400)
def sarima_forecast(ts):
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    result = model.fit(disp=False)
    return result.forecast(steps=1)[0]

def assign_weather_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Summer"
    elif month in [6, 7, 8]: return "Monsoon"
    else: return "Autumn"

def assign_festival_season(date):
    m, d = date.month, date.day
    if (m == 1 and d <= 15): return "New Year"
    elif (m == 3 and 1 <= d <= 20): return "Holi"
    elif (m == 4 and 1 <= d <= 25): return "Eid"
    elif (m == 8 and 1 <= d <= 30): return "Independence Day"
    elif (m == 10 and 1 <= d <= 15): return "Dussehra"
    elif (m == 2 and 1 <= d <= 15): return "Diwali"
    elif (m == 12 and 1 <= d <= 25): return "Christmas"
    else: return "None"

# Load data
df = load_all_data()
if df.empty:
    st.error("No data found.")
    st.stop()

monthly = preprocess_monthly_data(df)
seasonal_df = assign_seasonal_tags(df)

all_products = df["Product"].unique()
all_festivals = ['New Year', 'Holi', 'Eid', 'Independence Day', 'Dussehra', 'Diwali', 'Christmas', 'None']

customer_ids = df["Customer ID"].dropna().unique()
selected_customer = st.sidebar.selectbox("Select Customer ID", customer_ids)
max_customers = st.sidebar.slider("Max Customers for SARIMA", 1, 50, 10)

st.sidebar.markdown(" Seasonal & Trend analysis")
st.sidebar.markdown("### Filters for Seasonal & Trend Analysis")

selected_products = st.sidebar.multiselect("Select Product(s)", all_products)
go_button = st.sidebar.button("Go")
if go_button:
    if selected_products:
        st.session_state.confirmed_products = selected_products
        st.sidebar.success("Products selected. Go to **Seasonal Sales Analysis** tab.")
    else:
        st.sidebar.warning("Please select at least one product before clicking Go.")
shared_selected_festivals = st.sidebar.multiselect("Select Festival(s)", all_festivals, default=[])
shared_date_range = st.sidebar.date_input(
    "Select Date Range (From - To)",
    value=(df["Date"].min().date(), df["Date"].max().date()),
    key="shared_date_range_sidebar" 
)
shared_selected_products = st.session_state.get("confirmed_products", [])

shared_date_range = st.sidebar.date_input(
    "Select Date Range (From - To)",
    value=(df["Date"].min().date(), df["Date"].max().date()),
    key="shared_date_range"
)
if isinstance(shared_date_range, tuple) and len(shared_date_range) == 2:
    shared_start_date, shared_end_date = shared_date_range
else:
    st.warning("Invalid date range selected.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard (All Customers)",
    "Customer Analysis",
    "Seasonal Sales Analysis",
    "Customer Next Prediction",
    "Product Wise Trend"
])
# TAB 1: Full Dashboard
with tab1:
    st.header("📦 Fast and Slow-Moving Items (All Customers)")

    product_sales = df.groupby("Product")["Quantity"].sum().reset_index()
    threshold = product_sales["Quantity"].median()
    product_sales["Category"] = product_sales["Quantity"].apply(lambda x: "Fast-Moving" if x >= threshold else "Slow-Moving")
    show_fast = st.sidebar.selectbox("Show Fast-Moving Items",["All"])
    show_slow = st.sidebar.selectbox("Show Slow-Moving Items",["All"])


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

    st.header(" Monthly Sales Trends (All Customers)")
    selected_product = st.sidebar.selectbox("Select Product for Monthly Analysis", df["Product"].unique())
    seasonal_df_product = monthly[monthly["Product"] == selected_product]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=seasonal_df_product, x="Month", y="Quantity", palette="coolwarm", ax=ax2)
    ax2.set_title(f"Monthly Sales Trend: {selected_product}")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    seasonal_weather = seasonal_df[seasonal_df["Product"] == selected_product].groupby("Weather_Season")["Quantity"].sum().reset_index()
    fig_season1, ax_season1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=seasonal_weather, x="Weather_Season", y="Quantity", palette="Set2", ax=ax_season1)
    ax_season1.set_title(f"{selected_product} Demand by Weather Season")
    st.pyplot(fig_season1)

    seasonal_festival = seasonal_df[seasonal_df["Product"] == selected_product].groupby("Festival_Season")["Quantity"].sum().reset_index()
    fig_season2, ax_season2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=seasonal_festival, x="Festival_Season", y="Quantity", palette="Accent", ax=ax_season2)
    ax_season2.set_title(f"{selected_product} Demand by Festival Season")
    st.pyplot(fig_season2)


# TAB 2: Per-Customer Analysis
with tab2:
    st.header(f"Purchase Pattern for Customer: {selected_customer}")
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
        heatmap_df = customer_data.copy()
        heatmap_df["Month"] = heatmap_df["Date"].dt.to_period("M").dt.to_timestamp()
        heatmap_data = heatmap_df.pivot_table(index="Product", columns="Month", values="Quantity", aggfunc="sum", fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, ax=ax4)
        st.pyplot(fig4)


# === TAB 3 ===
with tab3:
    st.header("🎉 Seasonal & Festival Product Trends")
    for prod in shared_selected_products:
        st.subheader(f"📦 Product: {prod}")
        product_df = seasonal_df[seasonal_df["Product"] == prod].copy()
        product_df = product_df[(product_df["Date"].dt.date >= shared_start_date) & (product_df["Date"].dt.date <= shared_end_date)]

        if not product_df.empty:
            weather_data = product_df.groupby("Weather_Season")["Quantity"].sum().reset_index()
            fig_w, ax_w = plt.subplots(figsize=(6, 3))
            sns.barplot(data=weather_data, x="Weather_Season", y="Quantity", palette="Paired", ax=ax_w)
            ax_w.set_title("By Weather Season")
            st.pyplot(fig_w)

            festival_data = product_df.groupby("Festival_Season")["Quantity"].sum()
            festival_data = festival_data.reindex(all_festivals, fill_value=0).reset_index()
            festival_data.columns = ["Festival_Season", "Quantity"]
            fig_f, ax_f = plt.subplots(figsize=(8, 3))
            sns.barplot(data=festival_data, x="Festival_Season", y="Quantity", palette="Set2", ax=ax_f)
            ax_f.set_title("By Festival Season (All)")
            plt.xticks(rotation=45)
            st.pyplot(fig_f)

            if shared_selected_festivals:
                for fest in shared_selected_festivals:
                    st.markdown(f"**📌 Festival: {fest}**")
                    filtered_fest_df = product_df[product_df["Festival_Season"] == fest]
                    if not filtered_fest_df.empty:
                        st.dataframe(filtered_fest_df[["Date", "Product", "Quantity", "Festival_Season"]])
                    else:
                        st.info(f"No sales data for **{prod}** during **{fest}** between {shared_start_date} and {shared_end_date}.")
        else:
            st.warning(f"No data available for **{prod}** from {shared_start_date} to {shared_end_date}. Showing fallback data.")
            fallback_df = seasonal_df[seasonal_df["Product"] == prod]
            if not fallback_df.empty:
                st.dataframe(fallback_df[["Date", "Product", "Quantity", "Festival_Season"]].head(10))

# === TAB 4 (Next Prediction & SARIMA) ===
with tab4:
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

# === TAB 5 ===
with tab5:
    st.header("📊 Product Wise Seasonal & Festival Trend Analysis")
    for product in shared_selected_products:
        st.subheader(f"📦 Product: {product}")
        df_prod = seasonal_df[seasonal_df["Product"] == product].copy()
        df_in_range = df_prod[(df_prod["Date"].dt.date >= shared_start_date) & (df_prod["Date"].dt.date <= shared_end_date)]

        if df_in_range.empty:
            st.warning(f"No data found for **{product}** between {shared_start_date} and {shared_end_date}. Showing data up to latest available.")
            df_in_range = df_prod[df_prod["Date"].dt.date <= shared_end_date]
            if df_in_range.empty:
                st.info(f"No data available at all for **{product}**.")
                continue

        weather_grp = df_in_range.groupby("Weather_Season")["Quantity"].sum().reset_index()
        fig_w5, ax_w5 = plt.subplots(figsize=(6, 3))
        sns.barplot(data=weather_grp, x="Weather_Season", y="Quantity", palette="pastel", ax=ax_w5)
        ax_w5.set_title(f"{product} - Weather Season Sales")
        st.pyplot(fig_w5)

        fest_grp = df_in_range.groupby("Festival_Season")["Quantity"].sum()
        fest_grp = fest_grp.reindex(all_festivals, fill_value=0).reset_index()
        fest_grp.columns = ["Festival_Season", "Quantity"]
        fig_f5, ax_f5 = plt.subplots(figsize=(8, 3))
        sns.barplot(data=fest_grp, x="Festival_Season", y="Quantity", palette="Set3", ax=ax_f5)
        ax_f5.set_title(f"{product} - Festival Season Sales")
        plt.xticks(rotation=45)
        st.pyplot(fig_f5)

        if shared_selected_festivals:
            for fest in shared_selected_festivals:
                st.markdown(f"**📌 {product} during {fest}**")
                fest_df = df_in_range[df_in_range["Festival_Season"] == fest]
                if not fest_df.empty:
                    st.dataframe(fest_df[["Date", "Product", "Quantity", "Festival_Season"]])
                else:
                    st.info(f"No data for **{product}** during **{fest}** in selected period.")
