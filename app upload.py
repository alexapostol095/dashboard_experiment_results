import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

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


# ─── Build aggregated revenue, margin & quantity DataFrames ───────────────────
def make_agg_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # 1) grab the *raw* column names from the mapping
    raw_group    = st.session_state.column_mappings["StrategyBoxName"]
    raw_pricechg = st.session_state.column_mappings.get("Price change")

    raw_ta = st.session_state.column_mappings[f"{metric} Test After"]
    raw_tb = st.session_state.column_mappings[f"{metric} Test Before"]
    raw_ca = st.session_state.column_mappings[f"{metric} Control After"]
    raw_cb = st.session_state.column_mappings[f"{metric} Control Before"]

    # 2) do the grouping & summing on the raw headers
    group_cols = [raw_group] + ([raw_pricechg] if raw_pricechg else [])
    grp = (
      df
      .groupby(group_cols, as_index=False)
      .agg({
         raw_ta: "sum",
         raw_tb: "sum",
         raw_ca: "sum",
         raw_cb: "sum",
      })
    )

    # 3) now rename *all* of them to the standardized names
    rename_map = {
      raw_ta: "Test After",
      raw_tb: "Test Before",
      raw_ca: "Control After",
      raw_cb: "Control Before",
      raw_group: "StrategyBoxName",
    }
    if raw_pricechg:
      rename_map[raw_pricechg] = "Price change"

    grp.rename(columns=rename_map, inplace=True)

    # 4) compute deltas & pct changes on the standardized columns
    grp["Change Test"]    = grp["Test After"]  - grp["Test Before"]
    grp["Change Control"] = grp["Control After"] - grp["Control Before"]
    grp["%Change Test"]   = ((grp["Change Test"]    / grp["Test Before"])  * 100).round(2)
    grp["%Change Control"]= ((grp["Change Control"] / grp["Control Before"]) * 100).round(2)

    return grp




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
page = st.sidebar.radio("Go to", ["📁 Data Setup", "🏠 Home", "📊 Per Product Performance"])

# --------------------- DATA SETUP PAGE ---------------------
if page == "📁 Data Setup":
    st.markdown("<h1 style='color: #12123B;'>Data Setup</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='color: #414168;'>Map the columns of your uploaded dataset to the required metrics before proceeding. Please use the search function within the dropdown boxes to speed up the process. Keep the following in mind:</p>
        <ul style='color: #414168;'>
            <li><b>StrategyBoxName/Sensitivity</b> and <b>PriceChange</b>
                <ul>
                    <li>Supplied in the table. StrategyBoxName may instead be named as Sensitivity. If Price change does not exist, leave it blank. </li>
                </ul>
            </li>
            <li><b>After</b> and <b>Before</b>
                <ul>
                    <li>Refers to the periods before and after the price changes. Columns post-experiment fall under <b>After</b> and columns pre-experiment fall under <b>Before</b>.</li>
                </ul>
            </li>
            <li><b>Test</b> and <b>Control</b>
                <ul>
                    <li>The test columns should relate to the location that exhibited price changes, whereas the control column relates to the location that stayed the same.</li>
                </ul>
            </li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # Initialize the mappings dict exactly once
    if "column_mappings" not in st.session_state:
        st.session_state.column_mappings = {
            "StrategyBoxName": "",
            "Price change": "",
            "Revenue Test After": "",
            "Revenue Test Before": "",
            "Revenue Control After": "",
            "Revenue Control Before": "",
            "Margin Test After": "",
            "Margin Test Before": "",
            "Margin Control After": "",
            "Margin Control Before": "",
            "Quantity Test After": "",
            "Quantity Test Before": "",
            "Quantity Control After": "",
            "Quantity Control Before": "",
        }

    # Show the raw data preview
    st.markdown("<h4 style='color: #414168;'>Preview of uploaded data:</h4>", unsafe_allow_html=True)
    st.dataframe(product_df.head(), use_container_width=True)

    # List of keys we need to map
    required_keys = list(st.session_state.column_mappings.keys())

    st.markdown(
            """
            <style>
            .stSelectbox {
                margin-top: -35px !important; /* Adjust this value to reduce the gap */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Create one selectbox per required key
    for key in required_keys:
        st.markdown(f"<strong style='color:#12123B'>{key}</strong>", unsafe_allow_html=True)
        st.session_state.column_mappings[key] = st.selectbox(
            "", 
            [""] + list(product_df.columns), 
            index=([""] + list(product_df.columns)).index(st.session_state.column_mappings[key]) 
                  if st.session_state.column_mappings[key] in product_df.columns 
                  else 0,
            key=f"mapping_{key}"
        )

    # When the user clicks Save, validate and either error or success
    if st.button("Save Mappings"):
        missing = [k for k,v in st.session_state.column_mappings.items() 
                   if v == "" and k != "Price change"]
        if missing:
            st.error("Please map all of: " + ", ".join(missing))
        else:
            st.write("✅ All required columns mapped!  You can now switch to Home or Per-Product views.")


# Function to create bar charts with rounded values
def create_bar_chart(df, column, title):
    df = df.copy()
    df[column] = df[column].round(2)  # Ensure values are rounded before plotting
    fig = px.bar(df, x="Price change", y=column, color="StrategyBoxName", 
                 title=title, text=df[column].astype(str) + '%')
    return fig

# --------------------- HOME PAGE ---------------------
if page == "🏠 Home":
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
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Test After'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test Before: €{revenue_df['Test Before'].sum():,.2f}</p>
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
                        <h3 style="margin: 0; color: #12123B;">€{revenue_df['Control After'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control Before: €{revenue_df['Control Before'].sum():,.2f}</p>
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
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Test After'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_test_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Test Before: €{margin_df['Test Before'].sum():,.2f}</p>
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
                        <h3 style="margin: 0; color: #12123B;">€{margin_df['Control After'].sum():,.2f}</h3>
                        <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_control_pct)}</p>
                    </div>
                    <p style="margin: 0; color: #414168;">Control Before: €{margin_df['Control Before'].sum():,.2f}</p>
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
    perf_df = rename_columns(product_df, st.session_state.column_mappings)
    perf_df["ProductId"] = perf_df["ProductId"].astype(str)



    st.markdown("<h1 style='color: #12123B; text-align: left;'>Per Product Performance</h1>", unsafe_allow_html=True)

    # Search Input
    st.markdown("<h4 style='color: #414168;'>Search for a Product by ID to Examine its Performance:</h4>", unsafe_allow_html=True)

    # Search Input
    search_query = st.text_input('')

    # Filter DataFrame based on search query
    if search_query:
        filtered_df = perf_df[perf_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
    else:
        filtered_df = perf_df

    # Display Filtered Data Table
    st.dataframe(filtered_df, use_container_width=True)

    # --------------------- PERFORMANCE METRICS ---------------------

    # Only show performance metrics if exactly one product is selected
    if len(filtered_df) == 1:
        # Extract the selected product data
        product = filtered_df.iloc[0]
        
        # Calculate Performance Metrics for Revenue, Margin, and Quantity
        def calculate_performance_metric_product(test_after, test_before, control_after, control_before):
            test_pct_product = round(((test_after - test_before) / test_before) * 100, 2)
            control_pct_product = round(((control_after - control_before) / control_after) * 100, 2)
            perf_diff_product = round(test_pct_product - control_pct_product, 2)
            return test_pct_product, control_pct_product, perf_diff_product

        # Get the metrics for each category
        revenue_test_pct_product, revenue_control_pct_product, revenue_perf_diff_product = calculate_performance_metric_product(
            product['Revenue Test After'], product['Revenue Test Before'], product['Revenue Control After'], product['Revenue Control Before']
        )
        
        margin_test_pct_product, margin_control_pct_product, margin_perf_diff_product = calculate_performance_metric_product(
            product['Margin Test After'], product['Margin Test Before'], product['Margin Control After'], product['Margin Control Before']
        )

        quantity_test_pct_product, quantity_control_pct_product, quantity_perf_diff_product = calculate_performance_metric_product(
            product['Quantity Test After'], product['Quantity Test Before'], product['Quantity Control After'], product['Quantity Control Before']
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
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Revenue Test After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: €{product['Revenue Test Before']:,.2f}</p>
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
                            <h3 style="margin: 0; color: #12123B;">€{product['Revenue Control After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(revenue_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: €{product['Revenue Control Before']:,.2f}</p>
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
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">€{product['Margin Test After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: €{product['Margin Test Before']:,.2f}</p>
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
                            <h3 style="margin: 0; color: #12123B;">€{product['Margin Control After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(margin_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: €{product['Margin Control Before']:,.2f}</p>
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
                        <h5 style="margin: 0; color: #414168;">Test After</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #12123B;">{product['Quantity Test After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_test_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Test Before: {product['Quantity Test Before']:,.2f}</p>
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
                            <h3 style="margin: 0; color: #12123B;">{product['Quantity Control After']:,.2f}</h3>
                            <p style="margin: 0; display: inline; color: #414168;"> % Change: {style_pct_change(quantity_control_pct_product)}</p>
                        </div>
                        <p style="margin: 0; color: #414168;">Control Before: {product['Quantity Control Before']:,.2f}</p>
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
        "Revenue":  "Revenue Test After",
        "Margin":   "Margin Test After",
        "Quantity": "Quantity Test After"
    }
    selected_column = column_map[selected_metric]

    # Build & round the top/bottom tables
    top_products    = perf_df.nlargest(top_x, selected_column).copy()
    bottom_products = perf_df.nsmallest(top_x, selected_column).copy()
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
    fig_top.update_xaxes(type="category")


    # Plot Bottom X
    fig_bottom = px.bar(
        bottom_products,
        x="ProductId",
        y=selected_column,
        title=f"Bottom {top_x} Products by {selected_metric}",
        text=selected_column
    )
    st.plotly_chart(fig_bottom, use_container_width=True)
    fig_bottom.update_xaxes(type="category")




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

    top_fig.update_xaxes(type="category")


    # Plot the bottom X products by performance change difference
    bottom_fig = px.bar(
        bottom_performance,
        x="ProductId",
        y="Performance Change Diff",
        title=f"Bottom {top_x} Products by Performance Change Difference ({selected_metric})",
        text='Performance Change Diff',
        labels={"Performance Change Diff": "Performance Change Difference (%)"},
    )
    bottom_fig.update_xaxes(type="category")


    # Display the bar plots
    st.plotly_chart(top_fig, use_container_width=True)
    st.plotly_chart(bottom_fig, use_container_width=True)






