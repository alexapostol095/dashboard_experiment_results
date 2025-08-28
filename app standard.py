import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

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
    return pd.read_csv(csv, dtype={"ProductId": str})

product_df = load_product_df(uploaded)


# --- Auto-map columns if not already mapped ---
if "column_mappings" not in st.session_state:
    required_columns = [
        "Revenue Test After", "Revenue Test Before",
        "Revenue Control After", "Revenue Control Before",
        "Margin Test After", "Margin Test Before",
        "Margin Control After", "Margin Control Before",
        "Quantity Test After", "Quantity Test Before",
        "Quantity Control After", "Quantity Control Before"
    ]
    # Try to find Sensitivity (optional)
    sensitivity_col = "Sensitivity" if "Sensitivity" in product_df.columns else None

    st.session_state.column_mappings = {}
    if sensitivity_col:
        st.session_state.column_mappings["Sensitivity"] = sensitivity_col
    for col in required_columns:
        st.session_state.column_mappings[col] = col


# ─── Build aggregated revenue, margin & quantity DataFrames ───────────────────
def make_agg_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Use Sensitivity if present in both mapping and DataFrame, otherwise no grouping
    raw_group = st.session_state.column_mappings.get("Sensitivity", None)
    if raw_group is not None and raw_group not in df.columns:
        raw_group = None

    raw_ta = st.session_state.column_mappings[f"{metric} Test After"]
    raw_tb = st.session_state.column_mappings[f"{metric} Test Before"]
    raw_ca = st.session_state.column_mappings[f"{metric} Control After"]
    raw_cb = st.session_state.column_mappings[f"{metric} Control Before"]

    group_cols = [raw_group] if raw_group else []
    grp = (
      df
      .groupby(group_cols, as_index=False) if group_cols else df
    )
    if group_cols:
        grp = grp.agg({
            raw_ta: "sum",
            raw_tb: "sum",
            raw_ca: "sum",
            raw_cb: "sum",
        })
    else:
        grp = pd.DataFrame([{
            raw_ta: df[raw_ta].sum(),
            raw_tb: df[raw_tb].sum(),
            raw_ca: df[raw_ca].sum(),
            raw_cb: df[raw_cb].sum(),
        }])

    # Rename columns
    rename_map = {
      raw_ta: "Test After",
      raw_tb: "Test Before",
      raw_ca: "Control After",
      raw_cb: "Control Before",
    }
    if raw_group:
        rename_map[raw_group] = "Sensitivity"

    grp.rename(columns=rename_map, inplace=True)

    # Compute deltas & pct changes
    grp["Change Test"]    = grp["Test After"]  - grp["Test Before"]
    grp["Change Control"] = grp["Control After"] - grp["Control Before"]
    grp["%Change Test"]   = ((grp["Change Test"]    / grp["Test Before"])  * 100).round(2)
    grp["%Change Control"]= ((grp["Change Control"] / grp["Control Before"]) * 100).round(2)

    return grp


def calculate_aggregated_performance(df, test_after_col, test_before_col, control_after_col, control_before_col):
    test_after = df[test_after_col].sum()
    test_before = df[test_before_col].sum()
    control_after = df[control_after_col].sum()
    control_before = df[control_before_col].sum()

    test_pct = round(((test_after - test_before) / test_before) * 100, 2) if test_before != 0 else 0.0
    control_pct = round(((control_after - control_before) / control_before) * 100, 2) if control_before != 0 else 0.0
    perf_diff = round(test_pct - control_pct, 2)
    return test_pct, control_pct, perf_diff, test_after, test_before, control_after, control_before



# Correct Test % Change Calculation
def compute_percentage_change(df, column_after, column_before):
    return round(((df[column_after].sum() - df[column_before].sum()) / df[column_before].sum()) * 100, 2)


# Function to display arrows based on performance
def performance_arrow(perf_diff):
    if perf_diff > 0:
        return f"<span style='color: green;'>{perf_diff:.2f}% better than Control</span>"
    elif perf_diff < 0:
        return f"<span style='color: red;'>{abs(perf_diff):.2f}% worse than Control</span>"
    else:
        return f"<span style='color: #12123B;'>No difference from Control</span>"

def rename_columns(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """
    Rename the columns of the DataFrame based on the user-provided mappings.
    """
    # Filter out any empty mappings
    valid_mappings = {key: value for key, value in column_mappings.items() if value}
    
    # Reverse the mapping to rename columns
    rename_mapping = {value: key for key, value in valid_mappings.items()}
    
    # Rename the columns in the DataFrame
    return df.rename(columns=rename_mapping) 

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
    color_col = "Sensitivity" if "Sensitivity" in df.columns else None
    fig = px.bar(
        df, 
        x="Sensitivity" if color_col else df.index, 
        y=column, 
        color=color_col, 
        title=title, 
        text=df[column].astype(str) + '%'
    )
    return fig

# --------------------- HOME PAGE ---------------------
if page == "🏠 Home":
    st.markdown("""
    <script>
    window.scrollTo(0, 0);
    </script>
""", unsafe_allow_html=True)
    
    if "column_mappings" not in st.session_state or any(value == "" for key, value in st.session_state.column_mappings.items() if key not in ["Price change"]):
        st.error("Please complete the Data Setup page and map all required columns (except 'Price change') before proceeding.")
        st.stop()

    # Create aggregated DataFrames
    revenue_df = make_agg_df(product_df, "Revenue")
    margin_df = make_agg_df(product_df, "Margin")
    quantity_df = make_agg_df(product_df, "Quantity")
    revenue_test_pct = compute_percentage_change(revenue_df, "Test After", "Test Before")
    margin_test_pct = compute_percentage_change(margin_df, "Test After", "Test Before")
    quantity_test_pct = compute_percentage_change(quantity_df, "Test After", "Test Before")

    revenue_control_pct = compute_percentage_change(revenue_df, "Control After", "Control Before")
    margin_control_pct = compute_percentage_change(margin_df, "Control After", "Control Before")
    quantity_control_pct = compute_percentage_change(quantity_df, "Control After", "Control Before")

    # Round percentage changes in dataframe
    for df in [revenue_df, margin_df, quantity_df]:
        df["%Change Test"] = df["%Change Test"].round(2)
        df["%Change Control"] = df["%Change Control"].round(2)

    # Calculate Performance Difference and Round
    revenue_perf_diff = round(revenue_test_pct - revenue_control_pct, 2)
    margin_perf_diff = round(margin_test_pct - margin_control_pct, 2)
    quantity_perf_diff = round(quantity_test_pct - quantity_control_pct, 2)

    # Proceed with the rest of the Home Page logic
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
                    <h5 style="margin: 0; color: #414168;">Test After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Test After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test Before: €{revenue_df['Test Before'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Control After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control Before: €{revenue_df['Control Before'].sum():,.0f}</p>
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
                    <h5 style="margin: 0; color: #414168;">Test After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Test After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test Before: €{margin_df['Test Before'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Control After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control Before: €{margin_df['Control Before'].sum():,.0f}</p>
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
                    <h5 style="margin: 0; color: #414168;">Test After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">{quantity_df['Test After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test Before: {quantity_df['Test Before'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h5 style="margin: 0; color: #414168;">Control After</h5>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #12123B;">{quantity_df['Control After'].sum():,.0f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control Before: {quantity_df['Control Before'].sum():,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        quantity_perf_diff = round(quantity_test_pct - quantity_control_pct, 2)
        st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(quantity_perf_diff)}</b></div>", unsafe_allow_html=True)

 

    # --- Add the Matplotlib Figure ---
    # Data for the three categories: Revenue, Margin, and Quantity (in percentage)
    categories = ['Revenue Change', 'Margin Change', 'Quantity Change']
    test_values = [revenue_test_pct, margin_test_pct, quantity_test_pct]
    control_values = [revenue_control_pct, margin_control_pct, quantity_control_pct]

    # Create a bar plot for the data
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tight_layout()

    # Set the background color of the figure
    fig.patch.set_facecolor('#F9F6F0')

    # Set the background color of the axes (plot area)
    ax.set_facecolor('#F9F6F0')

    # Width of bars
    bar_width = 0.35

    # Position of bars on x-axis
    index = range(len(categories))

    # Plot bars for Test and Control
    bars1 = ax.bar(index, test_values, bar_width, label='Test', color='#3C37FF')  # Dark blue for Test
    bars2 = ax.bar([i + bar_width for i in index], control_values, bar_width, label='Control', color='#12123B')  # Darker blue for Control

    # Add data labels inside the bars
    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{bar.get_height():.2f}%', 
                ha='center', va='center', fontsize=10, color='white')

    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{bar.get_height():.2f}%', 
                ha='center', va='center', fontsize=10, color='white')  # White text for contrast

    # Draw arrows to highlight the differences


    # Labeling
    ax.set_xlabel('Category', color='black')  # Black color for axis labels
    ax.set_ylabel('Percentage Change (%)', color='black')  # Black color for axis labels
    ax.set_title('Percentage Changes in Revenue, Margin, and Quantity', color='black')  # Black title for contrast
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(categories, color='black')  # Black category labels
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Dropdown for selecting the data table to display
    st.markdown("<br><br>", unsafe_allow_html=True)
    sensitivity_present = "Sensitivity" in st.session_state.column_mappings and st.session_state.column_mappings["Sensitivity"] in product_df.columns
    if sensitivity_present:
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
    else:
        pass  # No message shown if Sensitivity is missing
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)




if page == "📊 Per Product Performance":
    st.markdown("""
    <style>
    /* Expander label */
    .streamlit-expanderHeader {
        color: #12123B !important;  /* or use #414168 if you want a softer tone */
        font-weight: bold;
    }

    /* Label text for multiselects */
    label {
        color: #12123B !important;
    }

    /* Optional: change dropdown text color as well */
    .stMultiSelect div[data-baseweb="select"] {
        color: #12123B !important;
    }
    </style>
""", unsafe_allow_html=True)
    st.markdown("""
<style>
/* Target the summary tag of the expander, which holds the label text */
details > summary {
    color: #12123B !important;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


    perf_df = rename_columns(product_df, st.session_state.column_mappings)
    perf_df["ProductId"] = perf_df["ProductId"].astype(str)

    st.markdown("<h1 style='color: #12123B; text-align: left;'>Per Product Performance</h1>", unsafe_allow_html=True)

    # --------------------- FILTER SECTION ---------------------

    with st.expander("🔍 Filter Products by Attributes", expanded=True):
        filter_columns = st.multiselect("Select attributes to filter on", perf_df.columns.tolist())

        filters = {}
        for col in filter_columns:
            unique_vals = perf_df[col].dropna().unique()
            selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=col)
            if selected_vals:
                filters[col] = selected_vals

        # Apply all filters
        filtered_df = perf_df.copy()
        for col, vals in filters.items():
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(vals)]

    # Display Filtered Data Table
    st.dataframe(filtered_df, use_container_width=True)

    # --------------------- PERFORMANCE METRICS ---------------------
    if not filtered_df.empty:
        # Revenue
        rev_test_pct, rev_ctrl_pct, rev_diff, rev_test_after, rev_test_before, rev_ctrl_after, rev_ctrl_before = calculate_aggregated_performance(
            filtered_df, 'Revenue Test After', 'Revenue Test Before', 'Revenue Control After', 'Revenue Control Before'
        )

        # Margin
        mar_test_pct, mar_ctrl_pct, mar_diff, mar_test_after, mar_test_before, mar_ctrl_after, mar_ctrl_before = calculate_aggregated_performance(
            filtered_df, 'Margin Test After', 'Margin Test Before', 'Margin Control After', 'Margin Control Before'
        )

        # Quantity
        qty_test_pct, qty_ctrl_pct, qty_diff, qty_test_after, qty_test_before, qty_ctrl_after, qty_ctrl_before = calculate_aggregated_performance(
            filtered_df, 'Quantity Test After', 'Quantity Test Before', 'Quantity Control After', 'Quantity Control Before'
        )

        col1, col2, col3 = st.columns(3)

        # --- REVENUE METRIC ---
        with col1:
            st.markdown("<h2 class='column-title'>Revenue</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;">
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">€{rev_test_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(rev_test_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: €{rev_test_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px;">
                        <h5 style="margin: 0; color: #414168;">Control After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">€{rev_ctrl_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(rev_ctrl_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: €{rev_ctrl_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(rev_diff)}</b></div>", unsafe_allow_html=True)

        # --- MARGIN METRIC ---
        with col2:
            st.markdown("<h2 class='column-title'>Margin</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;">
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">€{mar_test_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(mar_test_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: €{mar_test_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px;">
                        <h5 style="margin: 0; color: #414168;">Control After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">€{mar_ctrl_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(mar_ctrl_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: €{mar_ctrl_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(mar_diff)}</b></div>", unsafe_allow_html=True)

        # --- QUANTITY METRIC ---
        with col3:
            st.markdown("<h2 class='column-title'>Quantity</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;">
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">{qty_test_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(qty_test_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: {qty_test_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            with st.container():
                st.markdown(f"""
                    <div style="background-color:#F9F6F0; padding: 15px; border-radius: 0px;">
                        <h5 style="margin: 0; color: #414168;">Control After</h5>
                        <div style="display: flex; justify-content: space-between;">
                            <h3 style="margin: 0; color: #12123B;">{qty_ctrl_after:,.0f}</h3>
                            <p style="margin: 0; color: #414168;">% Change: {style_pct_change(qty_ctrl_pct)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: {qty_ctrl_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: center;'><b style='font-size: 20px;'>{performance_arrow(qty_diff)}</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='color: #12123B;'>Build a Visual Comparison</h2>", unsafe_allow_html=True)

    agg_method = st.selectbox("Select Aggregation Method", ["Sum", "Average"])
    metric = st.selectbox("Select Metric", ["Quantity", "Revenue", "Margin"])

    # --- Group 1: Multi-column filter ---
    group1_cols = st.multiselect("Select Group 1 Columns", product_df.columns.tolist(), key="group1_cols")
    filters_group1 = {}
    for col in group1_cols:
        filter_vals = st.multiselect(f"Select values for Group 1 ({col})", sorted(product_df[col].dropna().unique().astype(str)), key=f"group1_{col}")
        if filter_vals:
            filters_group1[col] = filter_vals

    # --- Group 2: Multi-column filter ---
    group2_cols = st.multiselect("Select Group 2 Columns", product_df.columns.tolist(), key="group2_cols")
    filters_group2 = {}
    for col in group2_cols:
        filter_vals = st.multiselect(f"Select values for Group 2 ({col})", sorted(product_df[col].dropna().unique().astype(str)), key=f"group2_{col}")
        if filter_vals:
            filters_group2[col] = filter_vals

    # Filter subsets for Group 1 (multi-column)
    df_group1 = product_df.copy()
    for col, vals in filters_group1.items():
        df_group1 = df_group1[df_group1[col].astype(str).isin(vals)]
    # Filter subsets for Group 2 (multi-column)
    if filters_group2:
        df_group2 = product_df.copy()
        for col, vals in filters_group2.items():
            df_group2 = df_group2[df_group2[col].astype(str).isin(vals)]
    else:
        df_group2 = pd.DataFrame()  # Empty DataFrame if no Group 2 selected

    agg_func = np.sum if agg_method == "Sum" else np.mean
    col_map = st.session_state.column_mappings
    test_before_col = col_map.get(f"{metric} Test Before")
    test_after_col = col_map.get(f"{metric} Test After")
    control_before_col = col_map.get(f"{metric} Control Before")
    control_after_col = col_map.get(f"{metric} Control After")

    # Defensive check if any col is None (mapping missing)
    missing_cols = [c for c in [test_before_col, test_after_col, control_before_col, control_after_col] if c is None]
    if missing_cols:
        st.error(f"Missing column mapping(s) for: {', '.join(missing_cols)}. Please check Data Setup.")
    else:
        # Filter subsets for Group 1 (multi-column)
        df_group1 = product_df.copy()
        for col, vals in filters_group1.items():
            df_group1 = df_group1[df_group1[col].astype(str).isin(vals)]
        # Filter subsets for Group 2 (multi-column)
        if filters_group2:
            df_group2 = product_df.copy()
            for col, vals in filters_group2.items():
                df_group2 = df_group2[df_group2[col].astype(str).isin(vals)]
        else:
            df_group2 = pd.DataFrame()  # Empty DataFrame if no Group 2 selected

        # Aggregate
        g1_test_before = agg_func(df_group1[test_before_col])
        g1_test_after = agg_func(df_group1[test_after_col])
        g1_control_before = agg_func(df_group1[control_before_col])
        g1_control_after = agg_func(df_group1[control_after_col])

        g2_test_before = agg_func(df_group2[test_before_col]) if not df_group2.empty else None
        g2_test_after = agg_func(df_group2[test_after_col]) if not df_group2.empty else None
        g2_control_before = agg_func(df_group2[control_before_col]) if not df_group2.empty else None
        g2_control_after = agg_func(df_group2[control_after_col]) if not df_group2.empty else None

        # Build plot data (handle case when Group 2 is missing)
        plot_df = []
        plot_df.append({"Group": "Group 1", "Period": "Before Test", "Value": g1_test_before})
        plot_df.append({"Group": "Group 1", "Period": "After Test",  "Value": g1_test_after})
        plot_df.append({"Group": "Group 1", "Period": "Before Control", "Value": g1_control_before})
        plot_df.append({"Group": "Group 1", "Period": "After Control", "Value": g1_control_after})

        if not df_group2.empty:
            plot_df.append({"Group": "Group 2", "Period": "Before Test", "Value": g2_test_before})
            plot_df.append({"Group": "Group 2", "Period": "After Test",  "Value": g2_test_after})
            plot_df.append({"Group": "Group 2", "Period": "Before Control", "Value": g2_control_before})
            plot_df.append({"Group": "Group 2", "Period": "After Control", "Value": g2_control_after})

        plot_df = pd.DataFrame(plot_df)

        # Create two columns for side-by-side display of Test and Control
        col1, col2 = st.columns(2)

        # Plot for Test
        with col1:
            fig_test = px.line(
                plot_df[plot_df["Period"].str.contains("Test")],
                x="Period",
                y="Value",
                color="Group",
                markers=True,
                title=f"{agg_method} {metric} Test Comparison Before and After"
            )
            fig_test.update_layout(
                xaxis_title="Period",
                yaxis_title=metric,
                legend_title="Group"
            )
            st.plotly_chart(fig_test, use_container_width=True)

        # Plot for Control
        with col2:
            fig_control = px.line(
                plot_df[plot_df["Period"].str.contains("Control")],
                x="Period",
                y="Value",
                color="Group",
                markers=True,
                title=f"{agg_method} {metric} Control Comparison Before and After"
            )
            fig_control.update_layout(
                xaxis_title="Period",
                yaxis_title=metric,
                legend_title="Group"
            )
            st.plotly_chart(fig_control, use_container_width=True)






    # ─── Top Performers ─────────────────────────────────────────────────────────
    st.markdown("<h2 style='text-align: left; color: #12123B;'>Top Performers</h2>", unsafe_allow_html=True)

    # --- Metric selection for Top Performers (must be before any use of selected_metric) ---
    selected_metric = st.selectbox(
        "Select Metric for Top/Bottom Performers:",
        ["Revenue", "Margin", "Quantity"],
        key="top_perf_metric",
        help="Select the metric to display Top/Bottom performers by"
    )

    # --- Top Performers Filter Section ---
    with st.expander("🔍 Filter Top Performers by Attributes", expanded=False):
        filter_columns_top = st.multiselect("Select attributes to filter on", perf_df.columns.tolist(), key="top_perf_filter_cols")
        filters_top = {}
        for col in filter_columns_top:
            unique_vals = perf_df[col].dropna().unique()
            selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=f"top_perf_{col}")
            if selected_vals:
                filters_top[col] = selected_vals
        # Apply all filters
        filtered_perf_df = perf_df.copy()
        for col, vals in filters_top.items():
            filtered_perf_df = filtered_perf_df[filtered_perf_df[col].astype(str).isin(vals)]
    # Use filtered_perf_df for top/bottom products
    df_for_top = filtered_perf_df if 'filtered_perf_df' in locals() else perf_df

    # Top-X dropdown (this already persists by default)
    top_x = st.selectbox(
        "Select the Number of Top Products to Display:",
        [5, 10, 15, 20, 30],
        index=0
    )

    # Map to the right column in product_df
    column_map = {
        "Revenue":  "Revenue Test After",
        "Margin":   "Margin Test After",
        "Quantity": "Quantity Test After"
    }
    selected_column = column_map[selected_metric]

    # Build & round the top/bottom tables
    top_products    = df_for_top[df_for_top[selected_column] != 0].nlargest(top_x, selected_column).copy()
    bottom_products = df_for_top[df_for_top[selected_column] != 0].nsmallest(top_x, selected_column).copy()
    top_products[selected_column]    = top_products[selected_column].round(2)
    bottom_products[selected_column] = bottom_products[selected_column].round(2)

    top_products["ProductId"] = top_products["ProductId"].astype(str)
    bottom_products["ProductId"] = bottom_products["ProductId"].astype(str)


    # Plot Top X
    fig_top = px.bar(
        top_products,
        x="ProductId",
        y=selected_column,
        title=f"Top {top_x} Products by {selected_metric}",
        text=selected_column
    )
    fig_top.update_xaxes(type="category")

    st.plotly_chart(fig_top, use_container_width=True)


    # Plot Bottom X
    fig_bottom = px.bar(
        bottom_products,
        x="ProductId",
        y=selected_column,
        title=f"Bottom {top_x} Products by {selected_metric}",
        text=selected_column
    )
    fig_bottom.update_xaxes(type="category");

    st.plotly_chart(fig_bottom, use_container_width=True)




# Calculate performance percentage change based on the selected metric
    # Function to calculate performance percentage changes and the performance difference

    # Calculate performance for the entire dataset
    # Function to calculate performance percentage changes and the performance difference
    def calculate_performance_metric_product(test_after, test_before, control_after, control_before):
    # Check for zero in the denominator before calculating percentage change
        if test_before == 0:
            test_pct_product = 0  # You can set it to 0 or another default value
        else:
            test_pct_product = round(((test_after - test_before) / test_before) * 100, 2)
        
        if control_before == 0:
            control_pct_product = 0  # You can set it to 0 or another default value
        else:
            control_pct_product = round(((control_after - control_before) / control_before) * 100, 2)
        
        perf_diff_product = round(test_pct_product - control_pct_product, 2)
        
        return test_pct_product, control_pct_product, perf_diff_product

# Calculate performance for the entire dataset
    def calculate_performance_for_df(df, metric):
        test_pct_list = []
        control_pct_list = []
        perf_diff_list = []

        # Define the correct column names based on the selected metric
        if metric == "Revenue":
            test_after_col = 'Revenue Test After'
            test_before_col = 'Revenue Test Before'
            control_after_col = 'Revenue Control After'
            control_before_col = 'Revenue Control Before'
        elif metric == "Margin":
            test_after_col = 'Margin Test After'
            test_before_col = 'Margin Test Before'
            control_after_col = 'Margin Control After'
            control_before_col = 'Margin Control Before'
        else:  # Quantity
            test_after_col = 'Quantity Test After'
            test_before_col = 'Quantity Test Before'
            control_after_col = 'Quantity Control After'
            control_before_col = 'Quantity Control Before'

        # Calculate percentage change for each row
        for _, row in df.iterrows():
            test_pct, control_pct, perf_diff = calculate_performance_metric_product(
                row[test_after_col], row[test_before_col], row[control_after_col], row[control_before_col]
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
    performance_df = calculate_performance_for_df(perf_df, selected_metric)
    performance_df["ProductId"] = performance_df["ProductId"].astype(str)

    # Get the top and bottom X products based on the performance change difference
    top_performance = performance_df.nlargest(top_x, 'Performance Change Diff')
    bottom_performance = performance_df.nsmallest(top_x, 'Performance Change Diff')
    top_performance["ProductId"] = top_performance["ProductId"].astype(str)
    bottom_performance["ProductId"] = bottom_performance["ProductId"].astype(str)

    # Plot the top X products by performance change difference
    top_fig = px.bar(
        top_performance,
        x="ProductId",
        y="Performance Change Diff",
        title=f"Top {top_x} Products by Performance Change Percentage Difference ({selected_metric})",
        text='Performance Change Diff',
        labels={"Performance Change Diff": "Performance Change Percentage Difference (%)"}
    )

    top_fig.update_xaxes(type="category")


    # Plot the bottom X products by performance change difference
    bottom_fig = px.bar(
        bottom_performance,
        x="ProductId",
        y="Performance Change Diff",
        title=f"Bottom {top_x} Products by Performance Change Percentage Difference ({selected_metric})",
        text='Performance Change Diff',
        labels={"Performance Change Diff": "Performance Change Percentage Difference (%)"},
    )
    bottom_fig.update_xaxes(type="category")


    # Display the bar plots
    st.plotly_chart(top_fig, use_container_width=True)
    st.plotly_chart(bottom_fig, use_container_width=True)

    # ─── Outlier Analysis ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<h2 style='text-align: left; color: #12123B;'>Outlier Analysis</h2>", unsafe_allow_html=True)

    # Dropdown to select metric
    outlier_metric = st.selectbox("Select Metric for Outlier Analysis", ["Revenue", "Margin", "Quantity"])

    # Define column names based on selected metric
    metric_columns = {
        "Revenue": ("Revenue Control Before", "Revenue Control After", "Revenue Test Before", "Revenue Test After"),
        "Margin": ("Margin Control Before", "Margin Control After", "Margin Test Before", "Margin Test After"),
        "Quantity": ("Quantity Control Before", "Quantity Control After", "Quantity Test Before", "Quantity Test After")
    }
    ctrl_before_col, ctrl_after_col, test_before_col, test_after_col = metric_columns[outlier_metric]

    # List all metric-related columns (across all metrics)
    all_metric_cols = sum(metric_columns.values(), ())

    # Add columns to explicitly exclude
    excluded_cols = set(all_metric_cols) | {"Test % Change", "Control % Change", "Performance Change Diff"}

    # Identify attribute columns to allow in filter UI
    filterable_cols = [col for col in perf_df.columns if col not in excluded_cols and col != "ProductId"]

    # ─── Filter Section ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filter Outlier Table", expanded=False):
        filter_columns = st.multiselect("Select attributes to filter on", filterable_cols)

        filters = {}
        for col in filter_columns:
            unique_vals = perf_df[col].dropna().unique()
            selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=f"outlier_{col}")
            if selected_vals:
                filters[col] = selected_vals

        # Apply filters to perf_df
        filtered_perf_df = perf_df.copy()
        for col, vals in filters.items():
            filtered_perf_df = filtered_perf_df[filtered_perf_df[col].astype(str).isin(vals)]

    # Subset data for outlier analysis
    attribute_cols = [col for col in filtered_perf_df.columns if col not in excluded_cols and col != "ProductId"]
    outlier_df = filtered_perf_df[["ProductId"] + attribute_cols +
                                [ctrl_before_col, ctrl_after_col, test_before_col, test_after_col]].copy()

    # Drop removed columns if still present
    outlier_df.drop(columns=["Test % Change", "Control % Change", "Performance Change Diff"], errors="ignore", inplace=True)

    # Ensure ProductId is string
    outlier_df["ProductId"] = outlier_df["ProductId"].astype(str)

    # Compute deltas
    outlier_df["Control Δ"] = outlier_df[ctrl_after_col] - outlier_df[ctrl_before_col]
    outlier_df["Test Δ"] = outlier_df[test_after_col] - outlier_df[test_before_col]
    outlier_df["Δ of Δs"] = outlier_df["Test Δ"] - outlier_df["Control Δ"]

    # Round numeric columns
    delta_cols = [ctrl_before_col, ctrl_after_col, test_before_col, test_after_col, "Control Δ", "Test Δ", "Δ of Δs"]
    outlier_df[delta_cols] = outlier_df[delta_cols].round(2)

    # Sort by Δ of Δs ascending
    outlier_df.sort_values(by="Δ of Δs", ascending=True, inplace=True)

    # Style the Δ of Δs column only
   # Plain table (no Styler)
    outlier_df = outlier_df.round(2)
    st.dataframe(outlier_df, use_container_width=True)














