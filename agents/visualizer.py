
import os
import json
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. LLM CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = (
    os.getenv("GROQ-key")
)

if not GROQ_API_KEY:
    raise ValueError("Groq API key missing in .env")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ============================================================
# 2. VISUALIZER AGENT FACTORY
# ============================================================

def build_visualizer_agent(dataframe):

    # Read-only DataFrame snapshot
    df = dataframe.copy(deep=True)

    # Store generated visualization plans
    dashboard_plan = {
        "charts": [],
        "filters": []
    }

    VALID_CHARTS = [
        "bar",
        "line",
        "area",
        "pie",
        "scatter",
        "histogram",
        "box"
    ]

    VALID_AGGREGATIONS = [
        "sum",
        "mean",
        "median",
        "count"
    ]

    # ========================================================
    # 3. VISUALIZATION PLANNING TOOL
    # ========================================================

    @tool
    def create_visualization_plan(
        charts: list[dict],
        filters: list[str]
    ) -> str:
        """
        Create a dashboard with one or multiple Plotly charts.

        Each chart must contain:

        type: bar, line, area, pie, scatter, histogram, box

        x: X-axis column

        y: Y-axis numerical column

        title: Chart title

        aggregation: sum, mean, median, count

        color: Optional column used for grouping/color.

        top_n: Optional number of categories to display.

        bins: Optional number of histogram bins.

        time_grain: Optional day, week, month, quarter, year.

        insight: Short explanation of what the chart shows.

        filters: List of columns to expose as dashboard slicers.

        Example:

        {
            "type": "bar",
            "x": "Vendor Name",
            "y": "Invoice Amt.",
            "title": "Top Vendors by Invoice Amount",
            "aggregation": "sum",
            "color": null,
            "top_n": 10
        }

        Use this tool to create all charts required by
        the user's visualization request.
        """

        errors = []
        validated_charts = []

        if not charts:
            return "No charts were supplied."

        if len(charts) > 8:
            return "Maximum 8 charts allowed per dashboard."

        # Validate dashboard filters
        valid_filters = []

        for col in filters[:6]:

            if col not in df.columns:
                errors.append(f"Invalid filter column: {col}")
            else:
                valid_filters.append(col)

        # Validate each chart
        for index, chart in enumerate(charts):

            chart_type = chart.get("type")
            x = chart.get("x")
            y = chart.get("y")
            color = chart.get("color")

            if chart_type not in VALID_CHARTS:
                errors.append(
                    f"Chart {index + 1}: Invalid chart type."
                )
                continue

            required_columns = []

            if chart_type != "histogram":
                if x:
                    required_columns.append(x)

            else:
                if not x:
                    errors.append(
                        f"Chart {index + 1}: Histogram requires x."
                    )
                    continue

                required_columns.append(x)

            if chart_type in [
                "bar", "line", "area", "pie",
                "scatter", "box"
            ]:
                if not y:
                    errors.append(
                        f"Chart {index + 1}: {chart_type} requires y."
                    )
                    continue

                required_columns.append(y)

            if color:
                required_columns.append(color)

            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                errors.append(
                    f"Chart {index + 1}: Missing columns {missing}"
                )
                continue

            if y and chart_type != "histogram":
                if not pd.api.types.is_numeric_dtype(df[y]):
                    errors.append(
                        f"Chart {index + 1}: Y-axis must be numeric."
                    )
                    continue

            if chart_type in ["scatter", "histogram"]:
                if not pd.api.types.is_numeric_dtype(df[x]):
                    errors.append(
                        f"Chart {index + 1}: X-axis must be numeric."
                    )
                    continue

            if chart_type == "box":
                if x and not pd.api.types.is_numeric_dtype(df[y]):
                    errors.append(
                        f"Chart {index + 1}: Box plot Y-axis must be numeric."
                    )
                    continue

            aggregation = chart.get("aggregation", "sum")

            if aggregation not in VALID_AGGREGATIONS:
                errors.append(
                    f"Chart {index + 1}: Invalid aggregation."
                )
                continue

            if chart_type in ["scatter", "histogram", "box"]:
                aggregation = None

            top_n = chart.get("top_n", 15)
            bins = chart.get("bins", 30)

            if not isinstance(top_n, int) or not 1 <= top_n <= 100:
                errors.append(
                    f"Chart {index + 1}: top_n must be 1-100."
                )
                continue

            if not isinstance(bins, int) or not 5 <= bins <= 100:
                errors.append(
                    f"Chart {index + 1}: bins must be 5-100."
                )
                continue

            time_grain = chart.get("time_grain")

            if time_grain not in [
                None, "day", "week", "month",
                "quarter", "year"
            ]:
                errors.append(
                    f"Chart {index + 1}: Invalid time_grain."
                )
                continue

            validated_charts.append({
                "type": chart_type,
                "x": x,
                "y": y,
                "title": chart.get(
                    "title",
                    f"Chart {index + 1}"
                ),
                "aggregation": aggregation,
                "color": color,
                "top_n": top_n,
                "bins": bins,
                "time_grain": time_grain,
                "insight": chart.get("insight", "")
            })

        if errors:
            return json.dumps({
                "status": "error",
                "errors": errors
            })

        dashboard_plan["charts"] = validated_charts
        dashboard_plan["filters"] = list(dict.fromkeys(valid_filters))

        return json.dumps({
            "status": "success",
            "charts_created": len(validated_charts),
            "filters": valid_filters,
            "chart_titles": [
                c["title"] for c in validated_charts
            ]
        })


    # ========================================================
    # 4. DATASET INFORMATION TOOL
    # ========================================================

    @tool
    def get_visualization_context() -> str:
        """
        Get dataset columns, data types, record count,
        unique values and numerical columns.

        Use this before planning visualizations if
        column selection is uncertain.
        """

        context = {
            "total_records": len(df),
            "total_columns": len(df.columns),

            "columns": df.columns.tolist(),

            "numeric_columns": [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ],

            "date_columns": [
                c for c in df.columns
                if "date" in c.lower()
                or pd.api.types.is_datetime64_any_dtype(df[c])
            ],

            "categorical_columns": [
                c for c in df.columns
                if (
                    df[c].dtype == "object"
                    or isinstance(df[c].dtype, pd.StringDtype)
                    or df[c].dtype.name == "category"
                )
            ],

            "unique_counts": {
                c: int(df[c].nunique())
                for c in df.columns
            }
        }

        return json.dumps(context, indent=2, default=str)


    # ========================================================
    # 5. REGISTER TOOLS
    # ========================================================

    tools = [
        get_visualization_context,
        create_visualization_plan
    ]


    # ========================================================
    # 6. SYSTEM PROMPT
    # ========================================================

    SYSTEM_PROMPT = f"""
    You are the Visualizer Agent of a procurement
    data intelligence platform.

    Your job is to understand visualization requests and
    create meaningful, interactive Plotly dashboards.

    DATASET CONTEXT:

    {df.columns.tolist()}

    Always use these exact column names.
    Do not assume that a column exists without checking
    the actual DataFrame schema.

    YOUR RESPONSIBILITIES:

    1. Understand what the user wants to visualize.
    2. Select suitable chart types.
    3. Select appropriate columns.
    4. Decide whether one chart or multiple charts
       are required.
    5. Recommend useful dashboard filters.
    6. Create the dashboard using the visualization tool.

    CHART SELECTION GUIDELINES:

    BAR:
    Compare vendors, categories, currencies and expenditure.

    LINE:
    Show changes and trends over time.

    AREA:
    Show cumulative or changing volume over time.

    PIE:
    Show composition or percentage contribution.
    Avoid pie charts for too many categories.

    SCATTER:
    Show relationships between two numerical variables.

    HISTOGRAM:
    Show numerical distributions.

    BOX:
    Compare numerical distributions and spread.

    IMPORTANT RULES:

    - Always use actual dataset columns.
    - Never invent column names.
    - Never invent numerical results.
    - Use the context tool if column information is unclear.
    - Use multiple charts when they provide complementary
      information.
    - Do not create redundant charts.
    - Prefer readable titles and meaningful axis choices.
    - Use appropriate aggregations.
    - For trends, use date columns and suitable time grains.
    - For financial analysis, select meaningful monetary
      measures such as Invoice Amt. or Line Amount.
    - Recommend relevant filters such as Vendor Name,
      Currency, Type, Posting Date and PLACE OF SUPPLY.
    - Limit dashboards to 8 charts.
    - Never modify the original DataFrame.
    - Never generate or execute arbitrary Python code.

    OUTPUT:

    Use create_visualization_plan when the user asks
    for charts, plots, graphs or dashboards.

    You may create multiple charts in one tool call.

    Explain briefly why the selected charts are useful.
    Do not claim a chart was generated unless the tool
    confirms successful validation.
    """


    # ========================================================
    # 7. LANGGRAPH WORKFLOW
    # ========================================================

    model_with_tools = llm.bind_tools(tools)

    def visualizer_node(state: MessagesState):

        response = model_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"]
            ]
        )

        return {
            "messages": [response]
        }


    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(
        "visualizer",
        visualizer_node
    )

    graph_builder.add_node(
        "tools",
        ToolNode(tools)
    )

    graph_builder.add_edge(
        START,
        "visualizer"
    )

    graph_builder.add_conditional_edges(
        "visualizer",
        tools_condition
    )

    graph_builder.add_edge(
        "tools",
        "visualizer"
    )

    visualizer_graph = graph_builder.compile()


    # ========================================================
    # 8. PUBLIC INTERFACE
    # ========================================================

    def run_visualizer(user_request: str):

        # Clear previous dashboard plan
        dashboard_plan["charts"] = []
        dashboard_plan["filters"] = []

        response = visualizer_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_request
                    }
                ]
            },
            config={
                "recursion_limit": 20
            }
        )

        final_message = response["messages"][-1]

        return {
            "response": final_message.content,

            "visualization_plan": {
                "charts": dashboard_plan["charts"].copy(),
                "filters": dashboard_plan["filters"].copy()
            },

            "dataframe": df.copy(deep=True),

            "rows": len(df),
            "columns": len(df.columns)
        }

    return run_visualizer


# ============================================================
# 9. PLOTLY RENDERING ENGINE
# ============================================================

def build_plotly_figure(dataframe, chart):

    data = dataframe.copy()

    chart_type = chart["type"]
    x = chart["x"]
    y = chart["y"]
    color = chart.get("color")

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    if chart_type == "histogram":

        fig = px.histogram(
            data,
            x=x,
            color=color,
            nbins=chart.get("bins", 30),
            title=chart["title"]
        )

    # --------------------------------------------------------
    # SCATTER
    # --------------------------------------------------------

    elif chart_type == "scatter":

        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            title=chart["title"],
            hover_data=[
                c for c in [
                    "Vendor Name",
                    "PO No",
                    "Material Description"
                ]
                if c in data.columns
            ]
        )

    # --------------------------------------------------------
    # BOX PLOT
    # --------------------------------------------------------

    elif chart_type == "box":

        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=chart["title"],
            points="outliers"
        )

    # --------------------------------------------------------
    # AGGREGATED CHARTS
    # --------------------------------------------------------

    else:

        # Convert date strings for time-based charts
        if chart.get("time_grain"):

            date_values = pd.to_datetime(
                data[x],
                errors="coerce"
            )

            data = data.copy()
            data[x] = date_values

            data = data.dropna(subset=[x])

            grain = chart["time_grain"]

            if grain == "day":
                data[x] = data[x].dt.strftime("%Y-%m-%d")

            elif grain == "week":
                data[x] = (
                    data[x].dt.to_period("W")
                    .astype(str)
                )

            elif grain == "month":
                data[x] = data[x].dt.to_period("M").astype(str)

            elif grain == "quarter":
                data[x] = data[x].dt.to_period("Q").astype(str)

            elif grain == "year":
                data[x] = data[x].dt.year.astype(str)

        group_columns = [x]

        if color and color != x:
            group_columns.append(color)

        if chart["aggregation"] == "count":

            grouped = (
                data.groupby(
                    group_columns,
                    dropna=False,
                    observed=True
                )
                .size()
                .reset_index(name="_value")
            )

        else:

            grouped = (
                data.groupby(
                    group_columns,
                    dropna=False,
                    observed=True
                )[y]
                .agg(chart["aggregation"])
                .reset_index(name="_value")
            )

        # Limit category-heavy charts
        if chart_type in ["bar", "pie"] and not chart.get("time_grain"):

            if len(grouped) > chart["top_n"]:

                top_categories = (
                    grouped.groupby(x)["_value"]
                    .sum()
                    .nlargest(chart["top_n"])
                    .index
                )

                grouped = grouped[
                    grouped[x].isin(top_categories)
                ]

        if chart_type == "bar":

            fig = px.bar(
                grouped,
                x=x,
                y="_value",
                color=color if color and color != x else None,
                barmode="group",
                title=chart["title"]
            )

        elif chart_type == "line":

            fig = px.line(
                grouped,
                x=x,
                y="_value",
                color=color if color and color != x else None,
                markers=True,
                title=chart["title"]
            )

        elif chart_type == "area":

            fig = px.area(
                grouped,
                x=x,
                y="_value",
                color=color if color and color != x else None,
                title=chart["title"]
            )

        elif chart_type == "pie":

            fig = px.pie(
                grouped,
                names=x,
                values="_value",
                title=chart["title"]
            )

    # --------------------------------------------------------
    # COMMON FORMATTING
    # --------------------------------------------------------

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=30, r=30, t=65, b=40),
        legend_title_text="Category",
        hoverlabel=dict(namelength=-1)
    )

    return fig
