import streamlit as st
import pandas as pd
import plotly.express as px

# Set full-width layout
st.set_page_config(layout="wide", page_title="Price Sensitivity Dashboard")

st.markdown(
    """
    <style>
    /* Change background color */
    body, .stApp {
        background-color: #E8F0FF;
    }

    /* Change sidebar color */
    .stSidebar {
        background-color: #3C37FF !important;
    }

    /* Sidebar text color */
    .stSidebar div {
        color: white !important;
    }

    /* Change text color for main content */
    .stMarkdown, .stText, .stSubheader, .stMetric, .stTitle, .stHeader, .stTable {
        color: #E8F0FF !important;
    }

    /* Style buttons */
    .stButton>button {
        background-color: #3C37FF !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }

    /* Style metric boxes */
    .stMetric {
        color: #E8F0FF !important;
    }

    /* Custom square box style */
    .metric-box {
        width: 250px;
        height: 250px;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background-color: #F9F6F0;
        margin-bottom: 10px;
    }

    /* Custom title color for overall title and per column titles */
    .main-title, .column-title {
        color: #12123B !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load Aggregated Data
st.sidebar.markdown("### 📁 Upload per-product CSV")
uploaded = st.sidebar.file_uploader(
    "Upload instructed CSV",
    type="csv",
)

if not uploaded:
    st.sidebar.info("Please upload your per-product CSV to enable the dashboard")
    st.stop()

@st.cache_data
def load_product_df(csv) -> pd.DataFrame:
    return pd.read_csv(csv)

product_df = load_product_df(uploaded)


# ─── Build aggregated revenue, margin & quantity DataFrames ───────────────────
def make_agg_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    prefix is the text before ' Test 25' etc, e.g. 'Total Revenue', 'Total Margin', or 'Quantity'
    """
    # 1) Sum up each of the four columns, grouping by StrategyBoxName & Price change
    grp = df.groupby(["StrategyBoxName", "Price change"], as_index=False).agg({
        f"{prefix} Test 25":    "sum",
        f"{prefix} Control 25": "sum",
        f"{prefix} Test 24":    "sum",
        f"{prefix} Control 24": "sum",
    })
    # 2) Rename to the year-based names your UI expects
    grp.rename(columns={
        f"{prefix} Test 25":    "Test 2025",
        f"{prefix} Control 25": "Control 2025",
        f"{prefix} Test 24":    "Test 2024",
        f"{prefix} Control 24": "Control 2024",
    }, inplace=True)

    # 3) Compute absolute & percent changes
    grp["Change Test"]    = grp["Test 2025"]    - grp["Test 2024"]
    grp["Change Control"] = grp["Control 2025"] - grp["Control 2024"]
    grp["%Change Test"]    = ((grp["Change Test"]    / grp["Test 2024"])    * 100).round(2)
    grp["%Change Control"] = ((grp["Change Control"] / grp["Control 2024"]) * 100).round(2)

    return grp

revenue_df = make_agg_df(product_df, "Total Revenue")
margin_df  = make_agg_df(product_df, "Total Margin")
quantity_df= make_agg_df(product_df, "Quantity")

# Correct Test % Change Calculation
def compute_percentage_change(df, column_2025, column_2024):
    return round(((df[column_2025].sum() - df[column_2024].sum()) / df[column_2024].sum()) * 100, 2)

revenue_test_pct = compute_percentage_change(revenue_df, "Test 2025", "Test 2024")
margin_test_pct = compute_percentage_change(margin_df, "Test 2025", "Test 2024")
quantity_test_pct = compute_percentage_change(quantity_df, "Test 2025", "Test 2024")

revenue_control_pct = compute_percentage_change(revenue_df, "Control 2025", "Control 2024")
margin_control_pct = compute_percentage_change(margin_df, "Control 2025", "Control 2024")
quantity_control_pct = compute_percentage_change(quantity_df, "Control 2025", "Control 2024")

# Round percentage changes in dataframe
for df in [revenue_df, margin_df, quantity_df]:
    df["%Change Test"] = df["%Change Test"].round(2)
    df["%Change Control"] = df["%Change Control"].round(2)

# Calculate Performance Difference and Round
revenue_perf_diff = round(revenue_test_pct - revenue_control_pct, 2)
margin_perf_diff = round(margin_test_pct - margin_control_pct, 2)
quantity_perf_diff = round(quantity_test_pct - quantity_control_pct, 2)

# Function to display arrows based on performance
def performance_arrow(perf_diff):
    if perf_diff > 0:
        return f"<span style='color: green;'>{perf_diff:.2f}% better than Control</span>"
    elif perf_diff < 0:
        return f"<span style='color: red;'>{abs(perf_diff):.2f}% worse than Control</span>"
    else:
        return f"<span style='color: #12123B;'>No difference from Control</span>"
    

def style_pct_change(pct_change):
    color = "green" if pct_change >= 0 else "red"
    return f'<span style="color: {color};">{pct_change}%</span>'

# Sidebar Navigation
st.sidebar.title("🔍 Select a View")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Per Product Performance"])

# Function to create bar charts with rounded values
def create_bar_chart(df, column, title):
    df = df.copy()
    df[column] = df[column].round(2)  # Ensure values are rounded before plotting
    fig = px.bar(df, x="Price change", y=column, color="StrategyBoxName", 
                 title=title, text=df[column].astype(str) + '%')
    return fig

# --------------------- HOME PAGE ---------------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title';'>Price Experiment Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])  

    st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #E8F0FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

   
# --- COLUMN 1: REVENUE ---
    with col1:
        st.markdown("<h2 class='column-title'>Revenue</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Test 2025'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test 2024: €{revenue_df['Test 2024'].sum():,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Control 2025'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control 2024: €{revenue_df['Control 2024'].sum():,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        revenue_perf_diff = round(revenue_test_pct - revenue_control_pct, 2)
        st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(revenue_perf_diff)}</b></div>", unsafe_allow_html=True)

    # --- COLUMN 2: MARGIN ---
    with col2:
        st.markdown("<h2 class='column-title' style='text-align: left;'>Margin</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Test 2025'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test 2024: €{margin_df['Test 2024'].sum():,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Control 2025'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control 2024: €{margin_df['Control 2024'].sum():,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        margin_perf_diff = round(margin_test_pct - margin_control_pct, 2)
        st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(margin_perf_diff)}</b></div>", unsafe_allow_html=True)

    # --- COLUMN 3: QUANTITY ---
    with col3:
        st.markdown("<h2 class='column-title' style='text-align: left;'>Quantity</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">{quantity_df['Test 2025'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test 2024: {quantity_df['Test 2024'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">{quantity_df['Control 2025'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control 2024: {quantity_df['Control 2024'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        quantity_perf_diff = round(quantity_test_pct - quantity_control_pct, 2)
        st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(quantity_perf_diff)}</b></div>", unsafe_allow_html=True)

    # Dropdown for selecting the data table to display
    st.markdown("<br><br>", unsafe_allow_html=True)
    selected_metric = st.selectbox(
        "Select the metric data table to display:",
        ["Revenue", "Margin", "Quantity"],
        key="dropdown",
        help="Select one of the metrics to display the corresponding data table"
    )

    # Styling the dropdown text
    st.markdown("""
    <style>
    .stSelectbox label {
        color: #414168 !important;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)



    # Display the corresponding data table based on user selection
    if selected_metric == "Revenue":
        st.markdown(f"<h3 style='text-align: center; color: #12123B;'>Revenue Results Table</h3>", unsafe_allow_html=True)
        st.dataframe(revenue_df, use_container_width=True)
    elif selected_metric == "Margin":
        st.markdown(f"<h3 style='text-align: center; color: #12123B;'>Margin Results Table</h3>", unsafe_allow_html=True)
        st.dataframe(margin_df, use_container_width=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: #12123B;'>Quantity Results Table</h3>", unsafe_allow_html=True)
        st.dataframe(quantity_df, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)




if page == "📊 Per Product Performance":
    st.markdown("<h1 style='color: #12123B; text-align: left;'>Per Product Performance</h1>", unsafe_allow_html=True)

    # Search Input
    st.markdown("<h4 style='color: #414168;'>Search for a Product by ID to Examine its Performance:</h4>", unsafe_allow_html=True)

    # Search Input
    search_query = st.text_input('')

    # Filter DataFrame based on search query
    if search_query:
        filtered_df = product_df[product_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
    else:
        filtered_df = product_df

    # Display Filtered Data Table
    st.dataframe(filtered_df, use_container_width=True)

    # --------------------- PERFORMANCE METRICS ---------------------

    # Only show performance metrics if exactly one product is selected
    if len(filtered_df) == 1:
        # Extract the selected product data
        product = filtered_df.iloc[0]
        
        # Calculate Performance Metrics for Revenue, Margin, and Quantity
        def calculate_performance_metric_product(test_2025, test_2024, control_2025, control_2024):
            test_pct_product = round(((test_2025 - test_2024) / test_2024) * 100, 2)
            control_pct_product = round(((control_2025 - control_2024) / control_2024) * 100, 2)
            perf_diff_product = round(test_pct_product - control_pct_product, 2)
            return test_pct_product, control_pct_product, perf_diff_product

        # Get the metrics for each category
        revenue_test_pct_product, revenue_control_pct_product, revenue_perf_diff_product = calculate_performance_metric_product(
            product['Total Revenue Test 25'], product['Total Revenue Test 24'], product['Total Revenue Control 25'], product['Total Revenue Control 24']
        )
        
        margin_test_pct_product, margin_control_pct_product, margin_perf_diff_product = calculate_performance_metric_product(
            product['Total Margin Test 25'], product['Total Margin Test 24'], product['Total Margin Control 25'], product['Total Margin Control 24']
        )

        quantity_test_pct_product, quantity_control_pct_product, quantity_perf_diff_product = calculate_performance_metric_product(
            product['Quantity Test 25'], product['Quantity Test 24'], product['Quantity Control 25'], product['Quantity Control 24']
        )

        # Columns for displaying the results
        col1, col2, col3 = st.columns(3)

        # --- REVENUE METRIC ---
        with col1:
            st.markdown("<h2 class='column-title'>Revenue</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Total Revenue Test 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test 2024: €{product['Total Revenue Test 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Total Revenue Control 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control 2024: €{product['Total Revenue Control 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(revenue_perf_diff_product)}</b></div>", unsafe_allow_html=True)

            

        # --- MARGIN METRIC ---
        with col2:
            st.markdown("<h2 class='column-title'>Margin</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Total Margin Test 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test 2024: €{product['Total Margin Test 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Total Margin Control 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control 2024: €{product['Total Margin Control 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(margin_perf_diff_product)}</b></div>", unsafe_allow_html=True)

        # --- QUANTITY METRIC ---
        with col3:
            st.markdown("<h2 class='column-title'>Quantity</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Test 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">{product['Quantity Test 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test 2024: {product['Quantity Test 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h5 style="margin: 0; color: #414168;">Control 2025</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">{product['Quantity Control 25']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control 2024: {product['Quantity Control 24']:,.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(quantity_perf_diff_product)}</b></div>", unsafe_allow_html=True)

    # ─── Top Performers ─────────────────────────────────────────────────────────
    st.markdown("<h2 style='text-align: left; color: #12123B;'>Top Performers</h2>", unsafe_allow_html=True)

    # your custom CSS for buttons & selectbox
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #F9F6F0 !important;
        color: #12123B !important;
        border-radius: 0px !important;
        border: 1px solid #12123B !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        width: 100%;
        height: 50px;
        font-size: 16px;
        text-align: center;
    }
    .stButton>button:hover {
        background-color: #12123B !important;
        color: #F9F6F0 !important;
    }
    /* Style the selectbox label text color */
    .stSelectbox label {
        color: #414168 !important;
        font-size: 16px;
    }
    /* Style the selectbox options */
    .stSelectbox select {
        background-color: #F9F6F0 !important;
        color: #12123B !important;
        border: 1px solid #12123B !important;
        border-radius: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # persist metric choice in session state
    if "selected_metric" not in st.session_state:
        st.session_state.selected_metric = "Revenue"

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Revenue"):
            st.session_state.selected_metric = "Revenue"
    with col2:
        if st.button("Margin"):
            st.session_state.selected_metric = "Margin"
    with col3:
        if st.button("Quantity"):
            st.session_state.selected_metric = "Quantity"

    selected_metric = st.session_state.selected_metric
    st.markdown(f"**Selected metric:** {selected_metric}")

    # Top-X dropdown (this already persists by default)
    top_x = st.selectbox(
        "Select the Number of Top Products to Display:",
        [5, 10, 15, 20, 30],
        index=0
    )

    # Map to the right column in product_df
    column_map = {
        "Revenue":  "Total Revenue Test 25",
        "Margin":   "Total Margin Test 25",
        "Quantity": "Quantity Test 25"
    }
    selected_column = column_map[selected_metric]

    # Build & round the top/bottom tables
    top_products    = product_df.nlargest(top_x, selected_column).copy()
    bottom_products = product_df.nsmallest(top_x, selected_column).copy()
    top_products[selected_column]    = top_products[selected_column].round(2)
    bottom_products[selected_column] = bottom_products[selected_column].round(2)

    # Plot Top X
    fig_top = px.bar(
        top_products,
        x="ProductId",
        y=selected_column,
        title=f"Top {top_x} Products by {selected_metric}",
        text=selected_column
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # Plot Bottom X
    fig_bottom = px.bar(
        bottom_products,
        x="ProductId",
        y=selected_column,
        title=f"Bottom {top_x} Products by {selected_metric}",
        text=selected_column
    )
    st.plotly_chart(fig_bottom, use_container_width=True)



# Calculate performance percentage change based on the selected metric
    # Function to calculate performance percentage changes and the performance difference

    # Calculate performance for the entire dataset
    # Function to calculate performance percentage changes and the performance difference
    def calculate_performance_metric_product(test_2025, test_2024, control_2025, control_2024):
    # Check for zero in the denominator before calculating percentage change
        if test_2024 == 0:
            test_pct_product = 0  # You can set it to 0 or another default value
        else:
            test_pct_product = round(((test_2025 - test_2024) / test_2024) * 100, 2)
        
        if control_2024 == 0:
            control_pct_product = 0  # You can set it to 0 or another default value
        else:
            control_pct_product = round(((control_2025 - control_2024) / control_2024) * 100, 2)
        
        perf_diff_product = round(test_pct_product - control_pct_product, 2)
        
        return test_pct_product, control_pct_product, perf_diff_product

# Calculate performance for the entire dataset
    def calculate_performance_for_df(df, metric):
        test_pct_list = []
        control_pct_list = []
        perf_diff_list = []

        # Define the correct column names based on the selected metric
        if metric == "Revenue":
            test_2025_col = 'Total Revenue Test 25'
            test_2024_col = 'Total Revenue Test 24'
            control_2025_col = 'Total Revenue Control 25'
            control_2024_col = 'Total Revenue Control 24'
        elif metric == "Margin":
            test_2025_col = 'Total Margin Test 25'
            test_2024_col = 'Total Margin Test 24'
            control_2025_col = 'Total Margin Control 25'
            control_2024_col = 'Total Margin Control 24'
        else:  # Quantity
            test_2025_col = 'Quantity Test 25'
            test_2024_col = 'Quantity Test 24'
            control_2025_col = 'Quantity Control 25'
            control_2024_col = 'Quantity Control 24'

        # Calculate percentage change for each row
        for _, row in df.iterrows():
            test_pct, control_pct, perf_diff = calculate_performance_metric_product(
                row[test_2025_col], row[test_2024_col], row[control_2025_col], row[control_2024_col]
            )
            test_pct_list.append(test_pct)
            control_pct_list.append(control_pct)
            perf_diff_list.append(perf_diff)

        # Add the calculated metrics to the dataframe
        df['Test % Change'] = test_pct_list
        df['Control % Change'] = control_pct_list
        df['Performance Change Diff'] = perf_diff_list
        return df

    # Calculate the performance for the selected metric (Revenue, Margin, or Quantity)
    performance_df = calculate_performance_for_df(product_df, selected_metric)

    # Get the top and bottom X products based on the performance change difference
    top_performance = performance_df.nlargest(top_x, 'Performance Change Diff')
    bottom_performance = performance_df.nsmallest(top_x, 'Performance Change Diff')

    # Plot the top X products by performance change difference
    top_fig = px.bar(
        top_performance,
        x="ProductId",
        y="Performance Change Diff",
        title=f"Top {top_x} Products by Performance Change Difference ({selected_metric})",
        text='Performance Change Diff',
        labels={"Performance Change Diff": "Performance Change Difference (%)"}
    )

    # Plot the bottom X products by performance change difference
    bottom_fig = px.bar(
        bottom_performance,
        x="ProductId",
        y="Performance Change Diff",
        title=f"Bottom {top_x} Products by Performance Change Difference ({selected_metric})",
        text='Performance Change Diff',
        labels={"Performance Change Diff": "Performance Change Difference (%)"},
    )

    # Display the bar plots
    st.plotly_chart(top_fig, use_container_width=True)
    st.plotly_chart(bottom_fig, use_container_width=True)





    
