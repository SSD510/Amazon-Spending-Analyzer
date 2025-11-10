import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Amazon Spending Analyzer")
st.write("Upload your Amazon spending data to visualize your expenses over time.")

uploaded_file = st.file_uploader("Choose CSV file", type = "csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = [c.strip().lower().replace(" ". "_") for c in df.columns]
    price_col = = None
    for col in df.columns:
        if "total" in col of "charged" in col or "price" in col;
            price_col = col
            break
    if price_col is None:
        st.error("Could not detect price column in uploaded file.")
    else:
        df[price_col] = (
            df[price_col]
            .replace("[\&,]", "", regex=True)
            .astype(float)
        )
        total_spent = df[price_col].sum()
        st.metric("Total Spent", f"${total_spent:,.2f}")
        st.subheader("Top 10 Most Expensive Purchases")
        top_items = df.nlargest(10, price_col)[["title", price_col]]
        st.dataframe(top_items)

        if "category" in df.columns:
            cat_sum - df.groupby("category")[price_col.sum().sort_values(ascending=False)
                                             st.subheader("Spending by Category")
                                             st.bar_chart(cat_sum)
        else:
            st.info("No 'category' column found, skipping category breakdown.")
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            time_sum = df.groupby(df["order_date"].dt.to_period("M"))[price_col].sum()
            time_sum.index = time_sum.index.to_timestamp()
            st.subheader("Spending Over Time")
            st.line_chart(time_sum)