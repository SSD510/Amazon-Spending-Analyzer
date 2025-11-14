import streamlit as st
import pandas as pd

st.title("Amazon Spending Analyzer")
st.write("Upload your Amazon spending data to visualize your expenses over time.")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ----------------------------
    # Detect price column
    # ----------------------------
    price_col = None
    price_keywords = ["total", "charged", "price", "amount"]

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
        .replace(r"[^\d\.-]", "", regex=True)  # remove $, commas, etc.
        .replace("", "0")                      # blank → zero
        .astype(float)
    )

    # ----------------------------
    # Detect title column
    # ----------------------------
    title_candidates = [
        "title", "item_name", "item", "product_name",
        "description", "product_title", "order_item"
    ]

    title_col = None
    for col in df.columns:
        if col in title_candidates:
            title_col = col
            break

    # ----------------------------
    # Total spent metric
    # ----------------------------
    total_spent = df[price_col].sum()
    st.metric("Total Spent", f"${total_spent:,.2f}")

    # ----------------------------
    # Top purchases table
    # ----------------------------
    st.subheader("Top 10 Most Expensive Purchases")

    if title_col:
        top_items = df.nlargest(10, price_col)[[title_col, price_col]]
    else:
        st.warning("No suitable title column found — showing prices only.")
        top_items = df.nlargest(10, price_col)[[price_col]]

    st.dataframe(top_items)

    # ----------------------------
    # Category breakdown
    # ----------------------------
    if "category" in df.columns:
        cat_sum = df.groupby("category")[price_col].sum().sort_values(ascending=False)

        st.subheader("Spending by Category")
        st.bar_chart(cat_sum)
    else:
        st.info("No 'category' column found, skipping category breakdown.")

    # ----------------------------
    # Time breakdown
    # ----------------------------
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        if df["order_date"].notna().sum() == 0:
            st.warning("Could not parse dates from 'order_date' column.")
        else:
            time_sum = df.groupby(df["order_date"].dt.to_period("M"))[price_col].sum()
            time_sum.index = time_sum.index.to_timestamp()

            st.subheader("Spending Over Time")
            st.line_chart(time_sum)
    else:
        st.info("No 'order_date' column found, skipping time breakdown.")