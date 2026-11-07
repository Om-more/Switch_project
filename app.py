
import streamlit as st
import pandas as pd

from supabase_pull import fetch_all
from agents.visualizer import (
    build_visualizer_agent,
    build_plotly_figure
)

st.set_page_config(
    page_title="Procurement Intelligence",
    layout="wide"
)

st.title("Procurement Intelligence Dashboard")

# Fetch data
@st.cache_data(ttl=300)
def load_data():
    return fetch_all()

df = load_data()

st.write(f"Total Records: {len(df)}")

# Build Visualizer
if "visualizer" not in st.session_state:
    st.session_state.visualizer = build_visualizer_agent(df)

if "visualization_plan" not in st.session_state:
    st.session_state.visualization_plan = None

# User request
prompt = st.text_input(
    "What would you like to visualize?",
    placeholder="Show top 10 vendors by total expenditure"
)

if st.button("Generate Dashboard"):

    if prompt.strip():

        with st.spinner("Generating visualization plan..."):

            result = st.session_state.visualizer(prompt)

            st.session_state.visualization_plan = (
                result["visualization_plan"]
            )

# Render charts
plan = st.session_state.visualization_plan

if plan:

    st.subheader("Dashboard Filters")

    filtered_df = df.copy()

    # Apply filters
    for col in plan.get("filters", []):

        if col not in filtered_df.columns:
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            minimum = float(series.min())
            maximum = float(series.max())

            if minimum < maximum:
                selected = st.slider(
                    col,
                    minimum,
                    maximum,
                    (minimum, maximum)
                )
                filtered_df = filtered_df[
                    filtered_df[col].between(*selected)
                ]

        else:
            options = sorted(
                series.dropna().astype(str).unique().tolist()
            )

            if len(options) <= 50:
                selected = st.multiselect(
                    col,
                    options,
                    default=options
                )

                filtered_df = filtered_df[
                    filtered_df[col].astype(str).isin(selected)
                ]

    st.divider()
    st.subheader("Generated Visualizations")

    charts = plan.get("charts", [])

    if not charts:
        st.warning("No valid charts were generated.")
    else:
        for chart in charts:

            st.markdown(f"### {chart['title']}")

            fig = build_plotly_figure(
                filtered_df,
                chart
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
            else:
                st.warning(
                    f"Could not render: {chart['title']}"
                )

