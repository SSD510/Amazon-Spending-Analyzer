import streamlit as st
import pandas as pd

st.title("Amazon Spending Analyzer")
st.write("Upload your Amazon spending data to visualize your expenses over time.")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect price column

    price_col = None
    price_keywords = ["total", "charged", "price", "amount", "cost"]

    for col in df.columns:
        if any(k in col for k in price_keywords):
            price_col = col
            break

    if price_col is None:
        st.error("Could not detect a price column in your CSV.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Clean & convert price column
    df[price_col] = (
        df[price_col]
        .replace(r"[^\d\.-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )

 
    # Detect title column

    title_candidates = [
        "title", "item_name", "item", "product_name",
        "description", "product_title", "order_item"
    ]

    title_col = next((col for col in df.columns if col in title_candidates), None)

 
    # Display totals
 
    total_spent = df[price_col].sum()
    st.metric("Total Spent", f"${total_spent:,.2f}")

 
    # Top purchases
 
    st.subheader("Top 10 Most Expensive Purchases")

    if title_col:
        top_items = df.nlargest(10, price_col)[[title_col, price_col]]
    else:
        st.warning("No suitable title column found — showing prices only.")
        top_items = df.nlargest(10, price_col)[[price_col]]

    st.dataframe(top_items)

 
    # Category breakdown
 
    if "category" in df.columns:
        cat_sum = df.groupby("category")[price_col].sum().sort_values(ascending=False)
        st.subheader("Spending by Category")
        st.bar_chart(cat_sum)
    else:
        st.info("No 'category' column found, skipping category breakdown.")

 
    # Time breakdown (Tabs + Modes)
 
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        if df["order_date"].notna().sum() == 0:
            st.warning("Could not parse any valid dates from 'order_date'.")
            st.stop()

        df = df.dropna(subset=["order_date"])

        # Extract time components
        df["year"] = df["order_date"].dt.year
        df["month"] = df["order_date"].dt.to_period("M").astype(str)
        df["week"] = df["order_date"].dt.strftime("%Y-W%U")

        st.subheader("Time-Based Spending Analysis")

     
        # Tabs
     
        tab1, tab2, tab3, tab4 = st.tabs([
            "📅 Monthly", 
            "📆 Weekly", 
            "📈 Yearly", 
            "📊 Rolling Averages"
        ])

     
        # Year filter
     
        years = sorted(df["year"].unique())
        selected_year = st.selectbox("Filter by Year", ["All"] + years)

        if selected_year != "All":
            df_filtered = df[df["year"] == selected_year]
        else:
            df_filtered = df

     
        # MONTHLY TAB
     
        with tab1:
            st.write("### Monthly Spending Breakdown")

            monthly = (
                df_filtered.groupby(df_filtered["order_date"].dt.to_period("M"))[price_col]
                .sum()
                .sort_index()
            )
            monthly.index = monthly.index.to_timestamp()

            st.bar_chart(monthly)

            monthly_df = monthly.reset_index()
            monthly_df.columns = ["Month", "Total Spent"]
            st.dataframe(monthly_df)

     
        # WEEKLY TAB
     
        with tab2:
            st.write("### Weekly Spending Breakdown")

            weekly = df_filtered.groupby("week")[price_col].sum()
            st.bar_chart(weekly)

            weekly_df = weekly.reset_index()
            weekly_df.columns = ["Week", "Total Spent"]
            st.dataframe(weekly_df)

     
        # YEARLY TAB
     
        with tab3:
            st.write("### Yearly Spending Breakdown")

            yearly = df.groupby("year")[price_col].sum()

            st.bar_chart(yearly)

            yearly_df = yearly.reset_index()
            yearly_df.columns = ["Year", "Total Spent"]
            st.dataframe(yearly_df)

     
        # ROLLING AVERAGE TAB
     
        with tab4:
            st.write("### 3-Month Rolling Spending Average")

            monthly_full = (
                df.groupby(df["order_date"].dt.to_period("M"))[price_col]
                .sum()
                .sort_index()
            )
            monthly_full.index = monthly_full.index.to_timestamp()

            rolling_3 = monthly_full.rolling(3).mean()

            st.line_chart(rolling_3)

            rolling_df = rolling_3.reset_index()
            rolling_df.columns = ["Month", "3-Month Rolling Average"]
            st.dataframe(rolling_df)

    else:
        st.info("No 'order_date' column found, skipping time-based analytics.")
