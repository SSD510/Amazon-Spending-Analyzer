# Amazon Shopping Data — Full Streamlit Dashboard with Insights
# Run with:
#   streamlit run Amazon_Streamlit_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    layout="wide",
    page_title="Amazon Spending Dashboard",
    initial_sidebar_state="expanded",
)

st.title("📦 Amazon Spending Analyzer — Multi-Page Dashboard")
st.write(
    "Upload your **Amazon Order History ZIP** (from Amazon's 'Request My Data' page) "
    "to explore spending, behavior, carbon footprint, and whether Prime is worth it."
)

# -----------------------------
# CATEGORY INFERENCE
# -----------------------------
def infer_category_from_title(title: str) -> str:
    """Heuristic category inference based on product title text."""
    t = str(title).lower()

    # Pet stuff
    if any(k in t for k in ["tiki cat", "cat food", "dog food", "pet", "litter"]):
        return "Pet Supplies"

    # Electronics / phone accessories
    if any(k in t for k in ["iphone", "phone case", "screen protector", "ipad",
                            "charger", "usb", "bluetooth", "airpods", "adapter"]):
        return "Electronics / Phone Accessories"

    # Clothing
    if any(k in t for k in ["hanes", "shirt", "hoodie", "jeans", "pants", "shorts",
                            "jacket", "sock", "boxers", "t-shirt", "sweater"]):
        return "Clothing"

    # Groceries / food
    if any(k in t for k in ["whole foods", "produce", "sourdough", "organic",
                            "potato", "cucumber", "zucchini", "bread", "loaf",
                            "spinach", "kale", "apple", "banana"]):
        return "Groceries"

    # Arts & crafts / jewelry making
    if any(k in t for k in ["beads", "bracelet", "earrings", "necklace", "diy",
                            "craft", "organza", "pin backs", "polymer clay"]):
        return "Arts & Crafts"

    # Personal care / hygiene
    if any(k in t for k in ["soap", "shampoo", "conditioner", "lotion",
                            "deodorant", "body wash", "toothpaste", "skincare"]):
        return "Personal Care"

    # Home / lighting / decor
    if any(k in t for k in ["light", "lamp", "desk lamp", "clip on light", "bulb",
                            "curtain", "pillow", "blanket", "sheet set"]):
        return "Home & Lighting"

    # Health & fitness
    if any(k in t for k in ["vitamin", "protein", "supplement", "creatine",
                            "omega-3", "preworkout", "dumbbell", "kettlebell"]):
        return "Health & Fitness"

    return "Other"


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Amazon ZIP file", type=["zip"])

if not uploaded_file:
    st.info("⬆️ Upload your Amazon ZIP to begin.")
    st.stop()

# Read the file into bytes so we can cache based on the content
zip_bytes = uploaded_file.read()

# -----------------------------
# ZIP HELPERS (BYTES-BASED)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_zip_files(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        return z.namelist()


@st.cache_data(show_spinner=False)
def read_csv_from_zip(zip_bytes: bytes, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        with z.open(member) as f:
            try:
                df = pd.read_csv(f, dtype=str)
            except Exception:
                f.seek(0)
                df = pd.read_csv(
                    io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                    dtype=str,
                )
    return df


@st.cache_data(show_spinner=False)
def read_json_from_zip(zip_bytes: bytes, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        with z.open(member) as f:
            return pd.read_json(f)


@st.cache_data(show_spinner=False)
def read_xlsx_from_zip(zip_bytes: bytes, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        with z.open(member) as f:
            return pd.read_excel(f)


@st.cache_data(show_spinner=False)
def load_all_tables(zip_bytes: bytes):
    """Return a dict of DataFrames keyed by filename inside the ZIP."""
    files = list_zip_files(zip_bytes)
    csvs = [f for f in files if f.lower().endswith(".csv")]
    jsons = [f for f in files if f.lower().endswith(".json")]
    xlsx = [f for f in files if f.lower().endswith(".xlsx")]

    tables: dict[str, pd.DataFrame] = {}

    for c in csvs:
        try:
            tables[c] = read_csv_from_zip(zip_bytes, c)
        except Exception as e:
            st.warning(f"Failed to parse CSV {c}: {e}")

    for j in jsons:
        try:
            tables[j] = read_json_from_zip(zip_bytes, j)
        except Exception:
            # Some JSONs may be complex; skip quietly
            pass

    for x in xlsx:
        try:
            tables[x] = read_xlsx_from_zip(zip_bytes, x)
        except Exception:
            pass

    return tables


# -----------------------------
# NORMALIZATION UTILITIES
# -----------------------------
def safe_money(x):
    """Convert currency-like strings to float, return NaN on failure."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def safe_date_parse(s):
    if pd.isna(s):
        return pd.NaT
    if isinstance(s, (pd.Timestamp, datetime)):
        return pd.to_datetime(s)

    # Try common formats first
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue

    # Fallback: let pandas guess
    try:
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.NaT


# -----------------------------
# BUILD CANONICAL ORDER DATAFRAME
# -----------------------------
@st.cache_data(show_spinner=False)
def build_canonical_orders(tables: dict[str, pd.DataFrame]):
    """Return (orders_df, order_file_name) using heuristics on Amazon exports."""

    # 1) Try typical Retail.OrderHistory file
    candidates = [
        k
        for k in tables.keys()
        if (
            "retail.orderhistory" in k.lower()
            or "orderhistory" in k.lower()
            or ("orders" in k.lower() and k.lower().endswith(".csv"))
        )
    ]

    order_file = None
    for c in candidates:
        if "Retail.OrderHistory.1" in c:
            order_file = c
            break
    if order_file is None and candidates:
        order_file = candidates[0]

    orders = None
    if order_file is not None:
        orders = tables[order_file].copy()
    else:
        # fallback: literally anything with 'order' in name
        for name in tables:
            if "order" in name.lower() and name.lower().endswith(".csv"):
                orders = tables[name].copy()
                order_file = name
                break

    if orders is None:
        st.error(
            "Could not find an Order History CSV in the ZIP. "
            "Look for something like 'Retail.OrderHistory.1.csv' in your export."
        )
        return None, None

    df = orders.copy()

    # --- FLEXIBLE COLUMN RENAME ---
    colmap = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    # Order ID
    for key in ("order id", "orderid", "order number", "order_number", "order-id"):
        if key in colmap:
            rename_map[colmap[key]] = "order_id"
            break

    # Order / purchase date
    for key in ("purchase date", "order date", "order_date", "purchase-date", "orderdate", "date"):
        if key in colmap:
            rename_map[colmap[key]] = "order_date"
            break

    # Title / item name
    for key in ("title", "item title", "product title", "order item", "product_title", "product name"):
        if key in colmap:
            rename_map[colmap[key]] = "title"
            break

    # Price / subtotal (extended for your file)
    for key in (
        "shipment item subtotal",
        "item subtotal",
        "item subtotal:",
        "subtotal",
        "total owed",
        "unit price",
        "price",
        "item price",
        "amount",
        "item price(usd)",
        "purchase price",
    ):
        if key in colmap:
            rename_map[colmap[key]] = "item_price"
            break

    # Category-ish (if Amazon ever provides it)
    for key in ("category", "product type", "department", "asin category"):
        if key in colmap:
            rename_map[colmap[key]] = "category"
            break

    # Seller
    for key in ("seller", "seller name"):
        if key in colmap:
            rename_map[colmap[key]] = "seller"
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    # --- TYPES ---
    if "item_price" in df.columns:
        df["item_price"] = df["item_price"].apply(safe_money)
    else:
        df["item_price"] = np.nan

    if "order_date" in df.columns:
        df["order_date"] = df["order_date"].apply(safe_date_parse)
        # Normalize timezone: convert tz-aware datetimes to naive for safe comparison
        try:
            if pd.api.types.is_datetime64tz_dtype(df["order_date"]):
                df["order_date"] = df["order_date"].dt.tz_convert(None)
        except Exception:
            pass
    else:
        # attempt to detect any date column
        for c in df.columns:
            if "date" in c.lower():
                df["order_date"] = df[c].apply(safe_date_parse)
                try:
                    if pd.api.types.is_datetime64tz_dtype(df["order_date"]):
                        df["order_date"] = df["order_date"].dt.tz_convert(None)
                except Exception:
                    pass
                break
        if "order_date" not in df.columns:
            df["order_date"] = pd.NaT

    # Derived columns
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M")
    df["weekday"] = df["order_date"].dt.day_name()
    df["hour"] = df["order_date"].dt.hour

    # Title column
    if "title" not in df.columns:
        possible_title = [
            c
            for c in df.columns
            if "title" in c.lower() or "product" in c.lower() or "item" in c.lower()
        ]
        if possible_title:
            df["title"] = df[possible_title[0]]
        else:
            df["title"] = "Unknown"

    # If no real category, derive from title
    if "category" not in df.columns or df["category"].nunique(dropna=True) <= 1:
        df["category"] = df["title"].apply(infer_category_from_title)

    # Try to merge cart items for richer info (if present)
    cart_file = None
    for k in tables.keys():
        if "retail.cartitems" in k.lower():
            cart_file = k
            break
    if cart_file:
        cart = tables[cart_file].copy()
        cart_cols = {c.lower(): c for c in cart.columns}
        for key in ("order id", "orderid", "order_id", "order-number"):
            if key in cart_cols:
                cart = cart.rename(columns={cart_cols[key]: "order_id"})
                break
        if "order_id" in cart.columns and "order_id" in df.columns:
            try:
                df = pd.merge(df, cart, on="order_id", how="left", suffixes=("", "_cart"))
            except Exception:
                pass

    return df, order_file


# -----------------------------
# FILTERS PANEL (DATA SCIENTIST MODE)
# -----------------------------
def filters_panel(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters & Controls")

    # Date range
    min_date = df["order_date"].min() if "order_date" in df.columns else None
    max_date = df["order_date"].max() if "order_date" in df.columns else None

    date_range = None
    if min_date is not None and pd.notna(min_date) and max_date is not None and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
        )

    keyword = st.sidebar.text_input("Product keyword (in title)")
    category_options = ["All"]
    if "category" in df.columns:
        cats = sorted(df["category"].dropna().unique().tolist())
        category_options += cats
    category = st.sidebar.selectbox("Category", category_options)

    min_price = max_price = None
    if "item_price" in df.columns and df["item_price"].notna().any():
        pmin = float(df["item_price"].min(skipna=True))
        pmax = float(df["item_price"].max(skipna=True))
        min_price, max_price = st.sidebar.slider(
            "Price range",
            min_value=float(pmin),
            max_value=float(pmax),
            value=(float(pmin), float(pmax)),
        )

    df_f = df.copy()

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        if start and end:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_f = df_f[(df_f["order_date"] >= start_dt) & (df_f["order_date"] <= end_dt)]

    if keyword:
        df_f = df_f[df_f["title"].str.contains(keyword, case=False, na=False)]

    if category and category != "All" and "category" in df_f.columns:
        df_f = df_f[df_f["category"] == category]

    if "item_price" in df_f.columns and min_price is not None:
        df_f = df_f[(df_f["item_price"] >= min_price) & (df_f["item_price"] <= max_price)]

    return df_f


# -----------------------------
# PAGES (Overview, Categories, Items, Returns, Digital, Cart, Raw)
# -----------------------------
def overview_page(df: pd.DataFrame, tables: dict[str, pd.DataFrame]):
    st.title("Overview — Amazon Spending Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    total_spent = df["item_price"].sum(skipna=True) if "item_price" in df.columns else None
    total_orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
    first_order = df["order_date"].min()
    last_order = df["order_date"].max()

    c1.metric("Total spent", f"${total_spent:,.2f}" if total_spent is not None else "N/A")
    c2.metric("Total orders", int(total_orders))
    c3.metric("First order", str(first_order.date()) if pd.notna(first_order) else "N/A")
    c4.metric("Most recent order", str(last_order.date()) if pd.notna(last_order) else "N/A")

    st.markdown("---")
    st.subheader("Spending over time (monthly)")
    if "order_date" in df.columns and "item_price" in df.columns:
        ts = df.set_index("order_date")["item_price"].resample("M").sum().reset_index()
        fig = px.area(ts, x="order_date", y="item_price", title="Monthly spend")
        fig.update_layout(yaxis_title="USD", xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No order_date or item_price column to show time series.")

    st.subheader("Top categories & items")
    if "category" in df.columns:
        cat = (
            df.groupby("category")["item_price"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            cat.head(15),
            x="item_price",
            y="category",
            orientation="h",
            title="Top categories by spend",
        )
        fig.update_layout(xaxis_title="Total spend", yaxis_title="Category")
        st.plotly_chart(fig, use_container_width=True)

    top_items = (
        df.groupby("title")
        .agg(total_spend=("item_price", "sum"), times=("order_id", "count"))
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    st.write("Top 10 items by spend")
    st.dataframe(top_items.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Ordering time-of-day heatmap")
    if "hour" in df.columns:
        heat = df.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heat = heat.reindex(weekdays)
        fig_h = go.Figure(
            data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index)
        )
        fig_h.update_layout(
            title="Orders by weekday and hour",
            xaxis_title="Hour of day",
            yaxis_title="Weekday",
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.subheader("Returns snapshot")
    returns_file = next(
        (k for k in tables.keys() if "customerreturns" in k.lower() or "ordersreturned" in k.lower()),
        None,
    )
    if returns_file:
        returns = tables[returns_file].copy()
        st.write(f"Found returns dataset: `{returns_file}`")
        if "amount" in returns.columns:
            returns["amount"] = returns["amount"].apply(safe_money)
            st.metric("Total refunded", f"${returns['amount'].sum():,.2f}")
        st.dataframe(returns.head(), use_container_width=True)
    else:
        st.info("No returns dataset discovered in archive.")


def categories_page(df: pd.DataFrame):
    st.title("Categories — Deep Dive")
    if "category" not in df.columns:
        st.info("No category info present in your orders data.")
        return

    agg = (
        df.groupby("category")
        .agg(total_spend=("item_price", "sum"), count=("order_id", "nunique"))
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    st.dataframe(agg, use_container_width=True)

    fig = px.pie(
        agg.head(20),
        names="category",
        values="total_spend",
        title="Top categories share of spend",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Category trend (monthly)")
    trend = df.groupby(["month", "category"])["item_price"].sum().reset_index()
    trend["month"] = trend["month"].dt.to_timestamp()

    top_cats = agg.head(6)["category"].tolist()
    trend_top = trend[trend["category"].isin(top_cats)]

    fig2 = px.line(
        trend_top,
        x="month",
        y="item_price",
        color="category",
        title="Monthly spend for top categories",
    )
    fig2.update_layout(xaxis_title="Month", yaxis_title="Spend")
    st.plotly_chart(fig2, use_container_width=True)


def items_page(df: pd.DataFrame):
    st.title("Items — Repeat, Top, & Reorder Suggestions")
    st.write("Top items by spend and frequency")

    items = (
        df.groupby("title")
        .agg(total_spend=("item_price", "sum"), times=("order_id", "count"))
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    st.dataframe(items.head(200), use_container_width=True)

    st.subheader("Repeat purchase behavior")
    if "order_id" in df.columns and "order_date" in df.columns:
        intervals = []
        for title, g in df.sort_values("order_date").groupby("title"):
            dates = g["order_date"].dropna().sort_values().unique()
            if len(dates) > 1:
                diffs = pd.Series(dates).diff().dropna().dt.days
                if not diffs.empty:
                    intervals.append(
                        {
                            "title": title,
                            "median_days": float(diffs.median()),
                            "purchases": len(dates),
                        }
                    )
        if intervals:
            intervals_df = pd.DataFrame(intervals).sort_values("median_days")
            st.write("Items with the shortest median refill intervals:")
            st.dataframe(intervals_df.head(50), use_container_width=True)

    st.markdown("### Naive next-reorder suggestions")
    st.info(
        "Uses median refill interval per title to suggest next order date. "
        "This is intentionally simple — not a full forecast."
    )

    suggestions = []
    for title, g in df.groupby("title"):
        dates = g["order_date"].dropna().sort_values()
        if len(dates) > 1:
            diffs = pd.Series(dates).diff().dropna().dt.days
            if not diffs.empty:
                median = diffs.median()
                last = dates.max()
                suggestions.append(
                    {
                        "title": title,
                        "last_order": last,
                        "median_days": float(median),
                        "suggested_next": last + pd.Timedelta(days=int(median)),
                    }
                )
    if suggestions:
        sugg_df = pd.DataFrame(suggestions).sort_values("median_days")
        st.dataframe(sugg_df.head(30), use_container_width=True)


def returns_page(tables: dict[str, pd.DataFrame]):
    st.title("Returns & Refunds")
    returns_file = next(
        (k for k in tables.keys() if "customerreturns" in k.lower() or "ordersreturned" in k.lower()),
        None,
    )
    if not returns_file:
        st.info("No returns dataset found in archive.")
        return

    r = tables[returns_file].copy()
    st.write(f"Found `{returns_file}` with {len(r)} rows")
    st.dataframe(r.head(200), use_container_width=True)

    if "amount" in r.columns:
        r["amount"] = r["amount"].apply(safe_money)
        st.metric("Total refunded", f"${r['amount'].sum():,.2f}")


def digital_page(tables: dict[str, pd.DataFrame]):
    st.title("Digital Orders & Subscriptions")
    candidates = [k for k in tables.keys() if "digital" in k.lower()]
    if not candidates:
        st.info("No digital-order files found in the ZIP.")
        return

    for c in candidates:
        st.markdown(f"#### {c}")
        st.dataframe(tables[c].head(200), use_container_width=True)


def cart_history_page(tables: dict[str, pd.DataFrame]):
    st.title("Cart History & Wishlist")
    cart_file = next(
        (k for k in tables.keys() if "cartitems" in k.lower()),
        None,
    )
    if not cart_file:
        st.info("No cart history found in archive.")
        return

    cart = tables[cart_file].copy()
    st.write(f"Loaded `{cart_file}` with {len(cart)} rows")
    st.dataframe(cart.head(200), use_container_width=True)


def raw_tables_page(tables: dict[str, pd.DataFrame]):
    st.title("Raw Tables — Browse & Inspect")
    choice = st.selectbox("Choose a table", sorted(tables.keys()))
    st.write(f"Showing `{choice}`")
    st.dataframe(tables[choice].head(1000), use_container_width=True)


# -----------------------------
# INSIGHTS / CARBON FOOTPRINT PAGE
# -----------------------------
def insights_page(df: pd.DataFrame):
    st.title("🧠 Insights — Behavior, Prime Value, and Carbon Footprint")

    # Safety checks
    if "order_id" not in df.columns or "order_date" not in df.columns or "item_price" not in df.columns:
        st.warning("Required columns (order_id, order_date, item_price) missing for insights.")
        return

    # Basic metrics
    total_orders = df["order_id"].nunique()
    total_spend = df["item_price"].sum(skipna=True)

    st.subheader("📦 Overview Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Orders", total_orders)
    c2.metric("Total Spend", f"${total_spend:,.2f}")
    c3.metric("Avg Spend per Order", f"${(total_spend / total_orders):.2f}" if total_orders > 0 else "N/A")

    # Prime value
    st.markdown("---")
    st.subheader("💸 Amazon Prime — Is It Worth It on Shipping Alone?")

    PRIME_COST = 139
    NON_PRIME_SHIPPING_COST = 5.99  # conservative estimate per shipment
    shipping_savings = total_orders * NON_PRIME_SHIPPING_COST
    net_value = shipping_savings - PRIME_COST

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Shipping Savings", f"${shipping_savings:,.2f}")
    c2.metric("Prime Membership Cost", f"${PRIME_COST}")
    c3.metric("Net Value of Prime", f"${net_value:,.2f}")

    if net_value > 0:
        st.success("Prime likely **paid for itself** based on shipping savings alone.")
    else:
        st.info("Prime may **not have fully paid for itself** on shipping alone.")

    # Carbon footprint
    st.markdown("---")
    st.subheader("🌍 Estimated Carbon Footprint of Your Orders")

    EMISSIONS_PER_PACKAGE = 1.0  # kg CO2e, delivery
    PACKAGING_WASTE = 0.3        # kg CO2e, packaging
    per_order_emissions = EMISSIONS_PER_PACKAGE + PACKAGING_WASTE

    est_total_emissions = total_orders * per_order_emissions
    st.metric("Estimated CO₂ Footprint", f"{est_total_emissions:,.1f} kg CO₂e")
    st.caption(
        "Estimate based on average last-mile delivery and packaging emissions per shipment. "
        "This is a rough but useful approximation for personal impact."
    )

    # Monthly emissions chart
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly_orders = df_monthly.groupby("month")["order_id"].nunique().reset_index()
    monthly_orders["emissions"] = monthly_orders["order_id"] * per_order_emissions

    if not monthly_orders.empty:
        fig = px.area(
            monthly_orders,
            x="month",
            y="emissions",
            title="Monthly Estimated Carbon Footprint (kg CO₂e)",
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Estimated kg CO₂e")
        st.plotly_chart(fig, use_container_width=True)

    # Impulse index
    st.markdown("---")
    st.subheader("⚡ Impulse Buying Index")

    item_counts = df["title"].value_counts()
    repeat_items = (item_counts > 1).sum()
    impulse_items = (item_counts == 1).sum()
    impulse_score = impulse_items / len(item_counts) if len(item_counts) > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Items Bought Once", impulse_items)
    c2.metric("Items Reordered", repeat_items)
    c3.metric("Impulse Score", f"{impulse_score:.2f}")

    if impulse_score > 0.7:
        st.warning("High impulse buying: most items were purchased only once.")
    elif impulse_score < 0.4:
        st.success("Low impulse buying: you tend to buy repeat-use, practical items.")
    else:
        st.info("Moderate impulse buying behavior.")

    # Category dominance
    st.markdown("---")
    st.subheader("📊 Category Drivers of Your Spending")

    if "category" in df.columns:
        cat_spend = (
            df.groupby("category")["item_price"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        if not cat_spend.empty:
            fig2 = px.bar(
                cat_spend.head(10),
                x="item_price",
                y="category",
                orientation="h",
                title="Top Categories by Spend",
            )
            fig2.update_layout(xaxis_title="Total Spend", yaxis_title="Category")
            st.plotly_chart(fig2, use_container_width=True)

            st.write("### Your Top Life Categories by Spend:")
            top3 = cat_spend.head(3)["category"].tolist()
            for i, cat in enumerate(top3, start=1):
                st.write(f"- **#{i}:** {cat}")
    else:
        st.info("No category information available (inferred or raw).")

    # Time-of-day & weekday behavior
    st.markdown("---")
    st.subheader("🕒 When You Shop the Most")

    df_time = df.copy()
    df_time["hour"] = df_time["order_date"].dt.hour
    df_time["weekday"] = df_time["order_date"].dt.day_name()

    if df_time["hour"].notna().any():
        busiest_hour = int(df_time["hour"].mode()[0])
    else:
        busiest_hour = None

    if df_time["weekday"].notna().any():
        busiest_day = df_time["weekday"].mode()[0]
    else:
        busiest_day = None

    c1, c2 = st.columns(2)
    c1.metric("Most Frequent Shopping Hour", f"{busiest_hour}:00" if busiest_hour is not None else "N/A")
    c2.metric("Most Frequent Shopping Day", busiest_day if busiest_day is not None else "N/A")

    heat = df_time.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    # Reindex rows in weekday order if present
    heat = heat.reindex(weekdays)

    fig3 = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="Blues",
        )
    )
    fig3.update_layout(
        title="Order Frequency by Weekday and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Weekday",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Narrative summary
    st.markdown("---")
    st.subheader("📖 Your Amazon Story — Summary")

    behavior_label = (
        "high" if impulse_score > 0.7 else "moderate" if impulse_score > 0.4 else "low"
    )
    summary_text = f"""
You placed **{total_orders} orders**, spending an estimated **${total_spend:,.2f}**.

Your estimated carbon footprint from deliveries and packaging is **{est_total_emissions:,.1f} kg CO₂e**.

Based on an assumed non-Prime shipping cost of **$5.99** per order, your estimated shipping savings are **${shipping_savings:,.2f}**, meaning your Prime membership is **{"likely worth it" if net_value > 0 else "probably not justified"}** on shipping alone.

Your impulse score of **{impulse_score:.2f}** suggests **{behavior_label}** impulse-buying behavior.
"""
    st.info(summary_text)


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    with st.spinner("Parsing ZIP and loading tables..."):
        tables = load_all_tables(zip_bytes)

    st.sidebar.title("Amazon Analytics")
    st.sidebar.success(f"Detected {len(tables)} tables in ZIP")

    orders_df, order_file = build_canonical_orders(tables)
    if orders_df is None:
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.write("Order source file:")
    st.sidebar.code(order_file or "(unknown)")

    # Navigation
    page = st.sidebar.selectbox(
        "Page",
        [
            "Overview",
            "Categories",
            "Items",
            "Returns",
            "Digital",
            "Cart",
            "Insights",
            "Raw tables",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    filtered_df = filters_panel(orders_df)

    if page == "Overview":
        overview_page(filtered_df, tables)
    elif page == "Categories":
        categories_page(filtered_df)
    elif page == "Items":
        items_page(filtered_df)
    elif page == "Returns":
        returns_page(tables)
    elif page == "Digital":
        digital_page(tables)
    elif page == "Cart":
        cart_history_page(tables)
    elif page == "Insights":
        insights_page(filtered_df)
    elif page == "Raw tables":
        raw_tables_page(tables)

    st.sidebar.markdown("---")
    st.sidebar.write("Built for you by Roci 🛰️")


if __name__ == "__main__":
    main()