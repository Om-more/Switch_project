
import streamlit as st
import pandas as pd

from supabase_pull import fetch_all

from agents.visualizer import (
    build_visualizer_agent,
    build_plotly_figure
)


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Procurement Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Procurement Intelligence Platform")
st.caption(
    "AI-powered visualization using Groq GPT-OSS-120B"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data():

    return fetch_all()


if "df" not in st.session_state:

    with st.spinner("Fetching data from Supabase..."):

        st.session_state.df = load_data()


df = st.session_state.df

if df.empty:

    st.error("No procurement records found.")
    st.stop()


# ============================================================
# 3. INITIALIZE VISUALIZER AGENT
# ============================================================

if "visualizer" not in st.session_state:

    st.session_state.visualizer = build_visualizer_agent(df)


if "dashboard_plan" not in st.session_state:

    st.session_state.dashboard_plan = None

if "agent_response" not in st.session_state:

    st.session_state.agent_response = ""


# ============================================================
# 4. USER QUERY
# ============================================================

st.subheader("Ask your Visualizer Agent")

with st.form("visualization_form"):

    user_query = st.text_area(
        "What would you like to visualize?",
        placeholder=(
            "Example: Create a dashboard showing monthly "
            "procurement expenditure, top 10 suppliers, "
            "currency distribution and invoice amount trends."
        ),
        height=100
    )

    generate = st.form_submit_button(
        "Generate Dashboard",
        type="primary"
    )


if generate:

    if not user_query.strip():

        st.warning("Please enter a visualization request.")

    else:

        try:

            with st.spinner(
                "AI is planning your visualizations..."
            ):

                result = st.session_state.visualizer(
                    user_query
                )

                st.session_state.dashboard_plan = (
                    result["visualization_plan"]
                )

                st.session_state.agent_response = (
                    result["response"]
                )

        except Exception as e:

            st.error(f"Visualization failed: {str(e)}")


# ============================================================
# 5. DISPLAY AGENT RESPONSE
# ============================================================

if st.session_state.agent_response:

    st.subheader("Visualizer Agent Response")

    st.write(st.session_state.agent_response)


# ============================================================
# 6. DASHBOARD FILTERS / SLICERS
# ============================================================

plan = st.session_state.dashboard_plan

if plan and plan["charts"]:

    st.divider()

    st.header("Interactive Dashboard")

    filtered_df = df.copy()

    filters = plan["filters"]

    st.sidebar.header("Dashboard Filters")

    st.sidebar.caption(
        "These filters apply to all generated charts."
    )

    for col in filters:

        series = df[col]

        # ----------------------------------------------------
        # DATE FILTER
        # ----------------------------------------------------

        if (
            "date" in col.lower()
            or pd.api.types.is_datetime64_any_dtype(series)
        ):

            date_values = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid_dates = date_values.dropna()

            if valid_dates.empty:
                continue

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            selected_dates = st.sidebar.date_input(
                f"Select {col} range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"date_filter_{col}"
            )

            if (
                isinstance(selected_dates, (tuple, list))
                and len(selected_dates) == 2
            ):

                start_date, end_date = selected_dates

                filtered_df = filtered_df[
                    (date_values >= pd.Timestamp(start_date))
                    & (
                        date_values
                        < pd.Timestamp(end_date)
                        + pd.Timedelta(days=1)
                    )
                ]

        # ----------------------------------------------------
        # NUMERICAL FILTER
        # ----------------------------------------------------

        elif pd.api.types.is_numeric_dtype(series):

            minimum = float(series.min())
            maximum = float(series.max())

            if pd.isna(minimum) or pd.isna(maximum):
                continue

            if minimum < maximum:

                selected_range = st.sidebar.slider(
                    f"Select {col} range",
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum),
                    key=f"number_filter_{col}"
                )

                filtered_df = filtered_df[
                    filtered_df[col].between(
                        selected_range[0],
                        selected_range[1]
                    )
                ]

        # ----------------------------------------------------
        # CATEGORICAL FILTER
        # ----------------------------------------------------

        else:

            unique_values = (
                series.dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(unique_values) <= 50:

                options = sorted(unique_values)

                selected_values = st.sidebar.multiselect(
                    f"Select {col}",
                    options=options,
                    default=[],
                    key=f"category_filter_{col}"
                )

                if selected_values:

                    filtered_df = filtered_df[
                        filtered_df[col]
                        .astype(str)
                        .isin(selected_values)
                    ]

            else:

                search_value = st.sidebar.text_input(
                    f"Search {col}",
                    key=f"search_filter_{col}"
                )

                if search_value:

                    filtered_df = filtered_df[
                        filtered_df[col]
                        .astype(str)
                        .str.contains(
                            search_value,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]


    # ========================================================
    # 7. DASHBOARD KPI INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )

    col2.metric(
        "Available Charts",
        len(plan["charts"])
    )

    col3.metric(
        "Active Filter Columns",
        len(filters)
    )


    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} records."
    )


    # ========================================================
    # 8. RENDER MULTIPLE PLOTLY CHARTS
    # ========================================================

    st.divider()

    charts = plan["charts"]

    for i in range(0, len(charts), 2):

        columns = st.columns(2)

        for j, container in enumerate(columns):

            chart_index = i + j

            if chart_index >= len(charts):
                break

            chart = charts[chart_index]

            with container:

                st.subheader(chart["title"])

                if chart.get("insight"):

                    st.caption(chart["insight"])

                try:

                    fig = build_plotly_figure(
                        filtered_df,
                        chart
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"plotly_chart_{chart_index}"
                    )

                except Exception as e:

                    st.error(
                        f"Could not render {chart['title']}: {e}"
                    )


    # ========================================================
    # 9. EXPORT FILTERED DATA
    # ========================================================

    st.divider()

    st.subheader("Export")

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_procurement_data.csv",
        mime="text/csv"
    )


    # ========================================================
    # 10. DATA PREVIEW
    # ========================================================

    with st.expander("View filtered dataset"):

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=350
        )

elif plan is not None:

    st.warning(
        "The agent did not generate a valid dashboard. "
        "Try a more specific visualization request."
    )
