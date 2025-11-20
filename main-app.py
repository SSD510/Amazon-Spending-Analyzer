# Amazon Shopping Data — Streamlit Dashboard
# File: Amazon_Streamlit_Dashboard.py
# Drop this file into a virtualenv with: pip install streamlit pandas plotly openpyxl
# Run with: streamlit run Amazon_Streamlit_Dashboard.py

import streamlit as st
import pandas as pd
import zipfile
import io
import altair as alt  # replacing plotly for portability

# --- File Upload ---
st.header("Upload Your Amazon ZIP Export")
uploaded_file = st.file_uploader("Upload the .zip file you downloaded from Amazon", type=["zip"])

# Placeholder for data until uploaded
orders = None
refunds = None
digital = None

if uploaded_file:
    with zipfile.ZipFile(uploaded_file) as z:
        sheet_files = [f for f in z.namelist() if f.lower().endswith((".csv", ".xlsx"))]

        def load_sheet(file):
            with z.open(file) as f:
                if file.lower().endswith(".csv"):
                    return pd.read_csv(f)
                else:
                    return pd.read_excel(f)

        for file in sheet_files:
            name = file.lower()
            df = load_sheet(file)
            if "order" in name and orders is None:
                orders = df
            elif "refund" in name and refunds is None:
                refunds = df
            elif "digital" in name and digital is None:
                digital = df

    if orders is not None:
        orders['Order Date'] = pd.to_datetime(orders['Order Date'], errors='coerce')
        orders = orders.dropna(subset=['Order Date'])
        orders['Subtotal'] = pd.to_numeric(orders.get('Item Subtotal', 0), errors='coerce').fillna(0)
    else:
        st.error("Orders file not found in the uploaded ZIP.")
        st.stop()

# Continue with original app after this point
import pandas as pd
import numpy as np
import zipfile
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="My Amazon Insights", initial_sidebar_state='expanded')

# IMPORTANT: path to the ZIP the user uploaded in this session
ZIP_PATH = "/mnt/data/Your Orders.zip"

# -----------------------------
# UTILITIES
# -----------------------------
@st.cache_data(show_spinner=False)
def list_zip_files(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        return z.namelist()

@st.cache_data(show_spinner=False)
def read_csv_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(member) as f:
            # handle bytes -> text
            try:
                df = pd.read_csv(f, dtype=str)
            except Exception:
                f.seek(0)
                df = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8', errors='replace'))
            return df

@st.cache_data(show_spinner=False)
def read_json_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(member) as f:
            return pd.read_json(f)

@st.cache_data(show_spinner=False)
def load_all_tables(zip_path):
    """Return a dict of dataframes keyed by filename"""
    files = list_zip_files(zip_path)
    csvs = [f for f in files if f.lower().endswith('.csv')]
    jsons = [f for f in files if f.lower().endswith('.json')]
    xlsx = [f for f in files if f.lower().endswith('.xlsx')]

    tables = {}
    for c in csvs:
        try:
            df = read_csv_from_zip(zip_path, c)
            tables[c] = df
        except Exception as e:
            st.warning(f"Failed to parse CSV {c}: {e}")
    for j in jsons:
        try:
            df = read_json_from_zip(zip_path, j)
            tables[j] = df
        except Exception as e:
            st.info(f"Skipped JSON {j}: {e}")
    # xlsx reading (if present)
    for x in xlsx:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(x) as f:
                    df = pd.read_excel(f)
                    tables[x] = df
        except Exception as e:
            st.info(f"Skipped XLSX {x}: {e}")

    return tables

# Normalize money columns to numeric
def safe_money(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    s = s.replace('$', '').replace(',', '').strip()
    try:
        return float(s)
    except:
        return np.nan

# Try to parse dates robustly
def safe_date_parse(s):
    if pd.isna(s):
        return pd.NaT
    if isinstance(s, (pd.Timestamp, datetime)):
        return pd.to_datetime(s)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(s, infer_datetime_format=True, errors='coerce')
    except Exception:
        return pd.NaT

# -----------------------------
# DATA LOADING & MERGE
# -----------------------------
@st.cache_data(show_spinner=False)
def build_canonical_orders(tables: dict):
    # Heuristics to find order history
    candidates = [k for k in tables.keys() if 'Retail.OrderHistory' in k or 'OrderHistory' in k or 'Retail.Orders' in k and k.lower().endswith('.csv')]
    # fallback: search for file with 'OrderHistory' or 'Order' in filename
    if not candidates:
        candidates = [k for k in tables.keys() if 'orderhistory' in k.lower() or 'orders' in k.lower() and k.lower().endswith('.csv')]

    # Best guess
    order_file = None
    for c in candidates:
        if 'Retail.OrderHistory.1' in c:
            order_file = c
            break
    if not order_file and candidates:
        order_file = candidates[0]

    orders = None
    if order_file:
        orders = tables[order_file].copy()
    else:
        # try common-named CSVs
        for name in tables:
            if 'order' in name.lower() and name.lower().endswith('.csv'):
                orders = tables[name].copy()
                break

    if orders is None:
        st.error('Could not find order history CSV in archive. Please confirm the ZIP contains Retail.OrderHistory.1/Retail.OrderHistory.csv')
        return None, order_file

    # Normalize columns: try to find likely column names
    df = orders
    # common columns alternatives
    rename_map = {}
    colmap = {c.lower(): c for c in df.columns}
    # Order id
    for k in ('order id','orderid','order number','order_number','order-id'):
        if k in colmap:
            rename_map[colmap[k]] = 'order_id'
            break
    # order date
    for k in ('purchase date','order date','order_date','purchase-date','orderdate'):
        if k in colmap:
            rename_map[colmap[k]] = 'order_date'
            break
    # item title
    for k in ('title','item title','product title','order item','product_title'):
        if k in colmap:
            rename_map[colmap[k]] = 'title'
            break
    # price
    for k in ('item subtotal','price','item price','amount','item price(usd)','purchase price'):
        if k in colmap:
            rename_map[colmap[k]] = 'item_price'
            break
    # category
    for k in ('category','product type','department','asin category'):
        if k in colmap:
            rename_map[colmap[k]] = 'category'
            break
    # seller
    for k in ('seller','seller name'):
        if k in colmap:
            rename_map[colmap[k]] = 'seller'
            break

    df = df.rename(columns=rename_map)

    # Coerce types
    if 'item_price' in df.columns:
        df['item_price'] = df['item_price'].apply(safe_money)
    # parse dates
    if 'order_date' in df.columns:
        df['order_date'] = df['order_date'].apply(safe_date_parse)
    else:
        # attempt to find any date-like column
        for c in df.columns:
            if 'date' in c.lower():
                df['order_date'] = df[c].apply(safe_date_parse)
                break

    # Ensure at least one date exists
    if 'order_date' not in df.columns:
        df['order_date'] = pd.NaT

    # Add useful normalized fields
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.to_period('M')
    df['weekday'] = df['order_date'].dt.day_name()
    df['hour'] = df['order_date'].dt.hour

    # fill title
    if 'title' not in df.columns:
        possible_title = [c for c in df.columns if 'title' in c.lower() or 'product' in c.lower() or 'item' in c.lower()]
        if possible_title:
            df['title'] = df[possible_title[0]]
        else:
            df['title'] = 'Unknown'

    # If there is a separate 'CartItems' dataset with richer itemization, try to merge
    cart_file = None
    for k in tables.keys():
        if 'Retail.CartItems' in k:
            cart_file = k
            break
    if cart_file:
        cart = tables[cart_file].copy()
        # normalize minimal
        cart_cols = {c.lower(): c for c in cart.columns}
        # try to find order id in cart
        for k in ('order id','orderid','order_id','order-number'):
            if k in cart_cols:
                cart = cart.rename(columns={cart_cols[k]: 'order_id'})
                break
        # try to join if possible
        if 'order_id' in cart.columns and 'order_id' in df.columns:
            try:
                merged = pd.merge(df, cart, on='order_id', how='left', suffixes=('', '_cart'))
                df = merged
            except Exception:
                pass

    return df, order_file

# -----------------------------
# VISUALIZATIONS & CALCULATIONS
# -----------------------------

def overview_page(df, tables):
    st.title("Overview — Amazon Spending Dashboard")
    c1, c2, c3, c4 = st.columns(4)

    total_spent = df['item_price'].sum() if 'item_price' in df.columns else None
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    first_order = df['order_date'].min()
    last_order = df['order_date'].max()

    c1.metric("Total spent", f"${total_spent:,.2f}" if total_spent is not None else "N/A")
    c2.metric("Total orders", int(total_orders))
    c3.metric("First order", str(first_order.date()) if pd.notna(first_order) else "N/A")
    c4.metric("Last order", str(last_order.date()) if pd.notna(last_order) else "N/A")

    st.markdown("---")
    st.subheader("Spending over time")
    if 'order_date' in df.columns and 'item_price' in df.columns:
        ts = df.set_index('order_date').resample('M')['item_price'].sum().reset_index()
        fig = px.area(ts, x='order_date', y='item_price', title='Monthly spend')
        fig.update_layout(yaxis_title='USD')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No order_date or item_price column to show time series')

    st.subheader("Top categories & items")
    if 'category' in df.columns:
        cat = df.groupby('category')['item_price'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat.head(15), x='item_price', y='category', orientation='h', title='Top categories by spend')
        st.plotly_chart(fig, use_container_width=True)

    top_items = df.groupby('title').agg({'item_price':'sum','order_id':'count'}).rename(columns={'order_id':'count'}).sort_values('item_price', ascending=False).reset_index()
    st.write('Top 10 items by spend')
    st.dataframe(top_items.head(10))

    st.markdown("---")
    st.subheader('Ordering time-of-day heatmap')
    if 'hour' in df.columns:
        heat = df.groupby(['weekday','hour']).size().unstack(fill_value=0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        fig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index))
        fig.update_layout(title='Orders by weekday and hour', xaxis_title='Hour of day', yaxis_title='Weekday')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('Returns snapshot')
    returns_file = next((k for k in tables.keys() if 'CustomerReturns' in k or 'OrdersReturned' in k), None)
    if returns_file:
        returns = tables[returns_file]
        st.write(f"Found returns dataset: {returns_file}")
        if 'amount' in returns.columns:
            returns['amount'] = returns['amount'].apply(safe_money)
            st.metric('Total refunded', f"${returns['amount'].sum():,.2f}")
        st.dataframe(returns.head())
    else:
        st.info('No returns dataset discovered in archive')

# Filters for C-style exploration

def filters_panel(df):
    st.sidebar.header('Filters & Controls (Data Scientist Mode)')
    min_date = df['order_date'].min() if 'order_date' in df.columns else None
    max_date = df['order_date'].max() if 'order_date' in df.columns else None
    date_range = st.sidebar.date_input('Date range', value=(min_date.date() if min_date is not pd.NaT else None, max_date.date() if max_date is not pd.NaT else None))

    keyword = st.sidebar.text_input('Product keyword (title contains)')
    category_options = ['All'] + (sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else [])
    category = st.sidebar.selectbox('Category', category_options)

    min_price, max_price = None, None
    if 'item_price' in df.columns:
        pmin = float(df['item_price'].min(skipna=True)) if df['item_price'].notna().any() else 0.0
        pmax = float(df['item_price'].max(skipna=True)) if df['item_price'].notna().any() else 100.0
        min_price, max_price = st.sidebar.slider('Price range', min_value=float(pmin), max_value=float(pmax), value=(pmin, pmax))

    # apply filters
    df_f = df.copy()
    if date_range and len(date_range) == 2 and 'order_date' in df.columns:
        start, end = date_range
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_f = df_f[(df_f['order_date'] >= start_dt) & (df_f['order_date'] <= end_dt)]
    if keyword:
        df_f = df_f[df_f['title'].str.contains(keyword, case=False, na=False)]
    if category and category != 'All' and 'category' in df_f.columns:
        df_f = df_f[df_f['category'] == category]
    if 'item_price' in df_f.columns and min_price is not None:
        df_f = df_f[(df_f['item_price'] >= min_price) & (df_f['item_price'] <= max_price)]

    return df_f

# Detail pages

def categories_page(df):
    st.title('Categories — Deep Dive')
    if 'category' not in df.columns:
        st.info('No category info present in your orders data.')
        return
    agg = df.groupby('category').agg(total_spend=('item_price','sum'), count=('order_id','nunique')).sort_values('total_spend', ascending=False).reset_index()
    st.dataframe(agg)
    fig = px.pie(agg.head(20), names='category', values='total_spend', title='Top categories share')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('Category trend (monthly)')
    trend = df.groupby(['month','category'])['item_price'].sum().reset_index()
    trend['month'] = trend['month'].dt.to_timestamp()
    top_cats = agg.head(6)['category'].tolist()
    trend_top = trend[trend['category'].isin(top_cats)]
    fig2 = px.line(trend_top, x='month', y='item_price', color='category', title='Monthly spend for top categories')
    st.plotly_chart(fig2, use_container_width=True)


def items_page(df):
    st.title('Items — Repeat, Top, & Predictions')
    st.write('Top items by spend and frequency')
    items = df.groupby('title').agg(total_spend=('item_price','sum'), times=('order_id','count')).sort_values('total_spend', ascending=False).reset_index()
    st.dataframe(items.head(200))

    st.subheader('Repeat purchase behavior')
    if 'order_id' in df.columns and 'order_date' in df.columns:
        # approximate repeat intervals per title
        intervals = []
        for title, g in df.sort_values('order_date').groupby('title'):
            dates = g['order_date'].dropna().sort_values().unique()
            if len(dates) > 1:
                diffs = pd.Series(dates).diff().dropna().dt.days
                intervals.append({'title': title, 'median_days': float(diffs.median()), 'count': len(dates)})
        if intervals:
            intervals_df = pd.DataFrame(intervals).sort_values('median_days')
            st.dataframe(intervals_df.head(50))

    st.markdown('Forecast next reorder (very naive)')
    st.info('This uses median refill interval per title to suggest next date — not a sophisticated forecast.')
    # show top 10 suggestions
    suggestions = []
    for title, g in df.groupby('title'):
        dates = g['order_date'].dropna().sort_values()
        if len(dates) > 1:
            median = pd.Series(dates).diff().dropna().dt.days.median()
            last = dates.max()
            suggestions.append({'title': title, 'last_order': last, 'median_days': median, 'suggested_next': last + pd.Timedelta(days=int(median))})
    if suggestions:
        sugg_df = pd.DataFrame(suggestions).sort_values('median_days')
        st.dataframe(sugg_df.head(30))


def returns_page(tables):
    st.title('Returns & Refunds')
    returns_file = next((k for k in tables.keys() if 'CustomerReturns' in k or 'OrdersReturned' in k), None)
    if not returns_file:
        st.info('No returns dataset found in archive')
        return
    r = tables[returns_file]
    st.write(f'Found {returns_file} with {len(r)} rows')
    st.dataframe(r.head(200))

    if 'amount' in r.columns:
        r['amount'] = r['amount'].apply(safe_money)
        st.metric('Total refunded', f"${r['amount'].sum():,.2f}")


def digital_page(tables):
    st.title('Digital Orders & Subscriptions')
    candidates = [k for k in tables.keys() if 'digital' in k.lower()]
    if not candidates:
        st.info('No digital order files found')
        return
    for c in candidates:
        st.write(c)
        st.dataframe(tables[c].head())


def cart_history_page(tables):
    st.title('Cart History & Wishlist')
    cart_file = next((k for k in tables.keys() if 'CartItems' in k), None)
    if not cart_file:
        st.info('No cart history found')
        return
    cart = tables[cart_file]
    st.write(f'Loaded {cart_file} with {len(cart)} rows')
    st.dataframe(cart.head(200))

# -----------------------------
# APP LAYOUT
# -----------------------------

def main():
    st.sidebar.title('Amazon Analytics')
    st.sidebar.markdown('File: ' + ZIP_PATH)

    if not os.path.exists(ZIP_PATH):
        st.error(f'ZIP file not found at {ZIP_PATH}. Please place your Amazon ZIP at this path or change ZIP_PATH variable.')
        return

    with st.spinner('Loading tables from ZIP — this may take a few seconds...'):
        tables = load_all_tables(ZIP_PATH)
    st.sidebar.success(f'Found {len(tables)} tables')

    df, order_file = build_canonical_orders(tables)
    if df is None:
        st.stop()

    # Sidebar navigation
    page = st.sidebar.selectbox('Page', ['Overview','Categories','Items','Returns','Digital','Cart','Raw tables'])

    # Filters panel (data scientist mode) — always available
    st.sidebar.markdown('---')
    st.sidebar.markdown('Data Scientist Controls')
    df_filtered = filters_panel(df)

    # Render page
    if page == 'Overview':
        overview_page(df_filtered, tables)
    elif page == 'Categories':
        categories_page(df_filtered)
    elif page == 'Items':
        items_page(df_filtered)
    elif page == 'Returns':
        returns_page(tables)
    elif page == 'Digital':
        digital_page(tables)
    elif page == 'Cart':
        cart_history_page(tables)
    elif page == 'Raw tables':
        st.title('Raw Tables — browse and inspect')
        table_choice = st.selectbox('Table', sorted(tables.keys()))
        st.write(f'Showing {table_choice}')
        st.dataframe(tables[table_choice].head(1000))

    st.sidebar.markdown('---')
    st.sidebar.write('Built for you by Roci')

if __name__ == '__main__':
    main()
