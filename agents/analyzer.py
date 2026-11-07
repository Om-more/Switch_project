
import os
import json
import operator
import pandas as pd
import numpy as np

from typing import Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START
)

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
# 2. ANALYZER AGENT FACTORY
# ============================================================

def build_analyzer_agent(dataframe, known_anomalies=None):

    # Read-only snapshot of the current DataFrame.
    df = dataframe.copy(deep=True)

    anomalies_df = (
        known_anomalies.copy(deep=True)
        if known_anomalies is not None
        else None
    )

    analysis_log = []

    # --------------------------------------------------------
    # HELPER: VALIDATE COLUMNS
    # --------------------------------------------------------

    def validate_columns(columns):

        missing = [
            col for col in columns
            if col not in df.columns
        ]

        if missing:
            return f"Columns not found: {missing}"

        return None


    def log_analysis(description):

        analysis_log.append(description)


    # ========================================================
    # TOOL 1: INSPECT DATASET
    # ========================================================

    @tool
    def inspect_dataset() -> str:
        """
        Inspect the complete dataset structure.

        Returns dimensions, column names, data types,
        missing values, duplicate records and memory usage.

        Use this when beginning an investigation or when
        the structure of the dataset is unknown.
        """

        report = {
            "total_records": len(df),
            "total_columns": len(df.columns),

            "columns": df.columns.tolist(),

            "data_types": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },

            "missing_values": {
                col: int(value)
                for col, value in df.isnull().sum().items()
            },

            "duplicate_records": int(df.duplicated().sum()),

            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 ** 2),
                2
            )
        }

        log_analysis("Inspected dataset structure.")

        return json.dumps(report, indent=2, default=str)


    # ========================================================
    # TOOL 2: NUMERICAL ANALYSIS
    # ========================================================

    @tool
    def numerical_analysis(
        columns: list[str]
    ) -> str:
        """
        Perform statistical analysis on numeric columns.

        Calculates:
        count, mean, median, standard deviation,
        variance, minimum, maximum, quartiles,
        skewness and missing values.

        Example columns:
        ["Unit Price", "Line Amount", "Invoice Amt."]
        """

        error = validate_columns(columns)

        if error:
            return error

        results = {}

        for col in columns:

            if not pd.api.types.is_numeric_dtype(df[col]):
                results[col] = {
                    "error": "Column is not numeric."
                }
                continue

            series = df[col].dropna()

            if series.empty:
                results[col] = {
                    "error": "No valid numerical values."
                }
                continue

            results[col] = {
                "count": int(series.count()),
                "missing": int(df[col].isna().sum()),

                "mean": float(series.mean()),
                "median": float(series.median()),

                "std_deviation": float(series.std()),
                "variance": float(series.var()),

                "minimum": float(series.min()),
                "maximum": float(series.max()),

                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),

                "skewness": float(series.skew())
            }

        log_analysis(
            f"Performed numerical analysis on {columns}."
        )

        return json.dumps(results, indent=2, default=str)


    # ========================================================
    # TOOL 3: CATEGORICAL ANALYSIS
    # ========================================================

    @tool
    def categorical_analysis(
        column: str,
        top_n: int = 15
    ) -> str:
        """
        Analyze categorical columns.

        Returns:
        - Unique value count
        - Frequency distribution
        - Percentage distribution
        - Missing values

        Useful for Vendor Name, Currency, Type,
        PLACE OF SUPPLY and product categories.
        """

        error = validate_columns([column])

        if error:
            return error

        if top_n < 1 or top_n > 100:
            return "top_n must be between 1 and 100."

        counts = (
            df[column]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(dropna=False)
        )

        total = len(df)

        distribution = []

        for value, count in counts.head(top_n).items():

            distribution.append({
                "value": str(value),
                "count": int(count),
                "percentage": round(
                    (count / total) * 100, 2
                ) if total else 0
            })

        result = {
            "column": column,
            "unique_values": int(df[column].nunique()),
            "missing_values": int(df[column].isna().sum()),
            "distribution": distribution,
            "total_records": total
        }

        log_analysis(
            f"Analyzed categorical distribution of {column}."
        )

        return json.dumps(result, indent=2)


    # ========================================================
    # TOOL 4: GROUP ANALYSIS
    # ========================================================

    @tool
    def group_analysis(
        group_by: list[str],
        metric: str | None = None,
        aggregation: Literal[
            "sum",
            "mean",
            "median",
            "count",
            "min",
            "max"
        ] = "sum",
        top_n: int = 20
    ) -> str:
        """
        Group records by one or more categorical columns
        and calculate an aggregate metric.

        Examples:

        Compare total Invoice Amt. by Vendor Name.

        Compare average Unit Price by Material Description.

        Compare purchase count by PLACE OF SUPPLY.

        group_by = ["Vendor Name"]
        metric = "Invoice Amt."
        aggregation = "sum"
        """

        error = validate_columns(group_by)

        if error:
            return error

        if metric is not None:
            error = validate_columns([metric])

            if error:
                return error

        if top_n < 1 or top_n > 100:
            return "top_n must be between 1 and 100."

        if aggregation != "count" and metric is None:
            return "A metric column is required for this aggregation."

        if metric is not None:
            if aggregation in ["sum", "mean", "median", "min", "max"]:
                if not pd.api.types.is_numeric_dtype(df[metric]):
                    return f"{metric} must be numeric."

        grouped = df.groupby(
            group_by,
            dropna=False,
            observed=True
        )

        if aggregation == "count":

            if metric is None:
                result = grouped.size().reset_index(name="result")
            else:
                result = grouped[metric].count().reset_index(name="result")

        else:

            result = (
                grouped[metric]
                .agg(aggregation)
                .reset_index(name="result")
            )

        result = result.sort_values(
            "result",
            ascending=False
        ).head(top_n)

        output = {
            "grouping_columns": group_by,
            "metric": metric,
            "aggregation": aggregation,
            "total_groups": int(len(result)),
            "results": result.to_dict(orient="records")
        }

        log_analysis(
            f"Grouped {metric or 'records'} by {group_by} using {aggregation}."
        )

        return json.dumps(output, indent=2, default=str)


    # ========================================================
    # TOOL 5: TIME SERIES / TREND ANALYSIS
    # ========================================================

    @tool
    def trend_analysis(
        date_column: str,
        metric: str,
        frequency: Literal["D", "W", "ME", "QE", "YE"] = "ME",
        aggregation: Literal[
            "sum",
            "mean",
            "count",
            "median"
        ] = "sum"
    ) -> str:
        """
        Analyze how a numerical metric changes over time.

        frequency:
        D  = Daily
        W  = Weekly
        ME = Monthly
        QE = Quarterly
        YE = Yearly

        Examples:
        Analyze monthly Invoice Amt.
        Analyze quarterly purchase volume.

        The date column is converted to datetime
        before aggregation.
        """

        error = validate_columns([date_column, metric])

        if error:
            return error

        if not pd.api.types.is_numeric_dtype(df[metric]):
            return f"{metric} must be numeric."

        date_df = df[[date_column, metric]].copy()

        date_df[date_column] = pd.to_datetime(
            date_df[date_column],
            errors="coerce"
        )

        date_df = date_df.dropna(subset=[date_column])

        if date_df.empty:
            return "No valid dates available for trend analysis."

        result = (
            date_df
            .set_index(date_column)
            .resample(frequency)[metric]
            .agg(aggregation)
            .dropna()
        )

        if result.empty:
            return "No valid trend results."

        trend_data = [
            {
                "period": str(index),
                "value": float(value)
            }
            for index, value in result.items()
        ]

        first_value = float(result.iloc[0])
        last_value = float(result.iloc[-1])

        percentage_change = None

        if first_value != 0:
            percentage_change = round(
                ((last_value - first_value) / abs(first_value)) * 100,
                2
            )

        output = {
            "date_column": date_column,
            "metric": metric,
            "frequency": frequency,
            "aggregation": aggregation,
            "periods": len(result),
            "first_period_value": first_value,
            "last_period_value": last_value,
            "percentage_change": percentage_change,
            "trend_data": trend_data[-60:]
        }

        log_analysis(
            f"Analyzed {metric} over time using {frequency} frequency."
        )

        return json.dumps(output, indent=2)


    # ========================================================
    # TOOL 6: OUTLIER DETECTION
    # ========================================================

    @tool
    def outlier_detection(
        column: str,
        method: Literal["IQR", "Z_SCORE"] = "IQR"
    ) -> str:
        """
        Detect statistically unusual numerical values.

        IQR:
        Uses Q1 - 1.5*IQR and Q3 + 1.5*IQR.

        Z_SCORE:
        Flags observations whose absolute Z-score
        exceeds 3.

        Returns thresholds, count, percentage
        and a sample of unusual records.

        Important: Statistical outliers are not automatically
        fraud or confirmed business anomalies.
        """

        error = validate_columns([column])

        if error:
            return error

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"{column} must be numeric."

        valid = df[column].dropna()

        if len(valid) < 3:
            return "Insufficient numerical records."

        if method == "IQR":

            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            mask = (
                (df[column] < lower)
                | (df[column] > upper)
            )

        elif method == "Z_SCORE":

            std = valid.std()

            if std == 0 or pd.isna(std):
                return "Cannot calculate Z-score: zero variance."

            mean = valid.mean()

            z_scores = (df[column] - mean) / std

            mask = z_scores.abs() > 3

            lower = mean - 3 * std
            upper = mean + 3 * std

        else:
            return "Unsupported outlier detection method."

        outliers = df.loc[mask]

        # Return limited evidence, not the entire dataset.
        sample_columns = [
            c for c in [
                "Vendor Code",
                "Vendor Name",
                "PO No",
                "Material Description",
                column,
                "Invoice Amt.",
                "Posting Date"
            ]
            if c in df.columns
        ]

        output = {
            "column": column,
            "method": method,
            "lower_threshold": float(lower),
            "upper_threshold": float(upper),
            "total_records": len(df),
            "outlier_count": len(outliers),
            "outlier_percentage": round(
                (len(outliers) / len(df)) * 100, 2
            ) if len(df) else 0,
            "sample_outliers": outliers[sample_columns]
                .head(15)
                .to_dict(orient="records")
        }

        log_analysis(
            f"Detected outliers in {column} using {method}."
        )

        return json.dumps(output, indent=2, default=str)


    # ========================================================
    # TOOL 7: CORRELATION ANALYSIS
    # ========================================================

    @tool
    def correlation_analysis(
        columns: list[str],
        method: Literal["pearson", "spearman"] = "pearson"
    ) -> str:
        """
        Calculate correlations between numerical columns.

        Pearson:
        Measures linear correlation.

        Spearman:
        Measures rank-based monotonic correlation.

        Correlation does not establish causation.
        """

        error = validate_columns(columns)

        if error:
            return error

        if len(columns) < 2:
            return "At least two columns are required."

        non_numeric = [
            c for c in columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]

        if non_numeric:
            return f"Non-numeric columns supplied: {non_numeric}"

        matrix = df[columns].corr(
            method=method
        )

        output = {
            "method": method,
            "correlation_matrix": matrix.round(4).to_dict(),
            "strong_relationships": []
        }

        for i, col_a in enumerate(columns):

            for col_b in columns[i + 1:]:

                value = matrix.loc[col_a, col_b]

                if pd.notna(value) and abs(value) >= 0.5:

                    output["strong_relationships"].append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": round(float(value), 4)
                    })

        log_analysis(
            f"Calculated {method} correlations for {columns}."
        )

        return json.dumps(output, indent=2, default=str)


    # ========================================================
    # TOOL 8: KNOWN ANOMALY INVESTIGATION
    # ========================================================

    @tool
    def investigate_known_anomalies() -> str:
        """
        Cross-reference the current procurement DataFrame
        against Known_anomalies.xlsx.

        Uses Vendor Code and PO No. as matching keys,
        and Material Description when available.

        Returns matching anomaly categories and records.

        If the anomaly reference file is unavailable,
        report that limitation.
        """

        if anomalies_df is None:
            return "Known anomaly reference data was not supplied."

        required = [
            "Vendor Code",
            "PO No",
            "Known Anomaly"
        ]

        missing = [
            col for col in required
            if col not in anomalies_df.columns
        ]

        if missing:
            return f"Anomaly reference missing columns: {missing}"

        join_keys = ["Vendor Code", "PO No"]

        if "Material Description" in df.columns and \
           "Material Description" in anomalies_df.columns:

            join_keys.append("Material Description")

        current = df.copy()
        reference = anomalies_df.copy()

        # Normalize key values to avoid simple whitespace issues.
        for col in join_keys:

            current[col] = (
                current[col].astype("string").str.strip()
            )

            reference[col] = (
                reference[col].astype("string").str.strip()
            )

        reference = reference.drop_duplicates(
            subset=join_keys + ["Known Anomaly"]
        )

        matched = current.merge(
            reference[join_keys + ["Known Anomaly"]].drop_duplicates(),
            on=join_keys,
            how="inner"
        )

        if matched.empty:
            return (
                "No matches found between the current dataset "
                "and the known anomaly reference."
            )

        category_counts = (
            matched["Known Anomaly"]
            .value_counts()
            .to_dict()
        )

        output = {
            "reference_records": len(anomalies_df),
            "matched_records": len(matched),
            "anomaly_categories": category_counts,
            "matched_examples": matched[
                join_keys + ["Known Anomaly"]
            ].head(20).to_dict(orient="records")
        }

        log_analysis(
            "Cross-referenced procurement records with known anomalies."
        )

        return json.dumps(output, indent=2, default=str)


    # ========================================================
    # 3. REGISTER TOOLS
    # ========================================================

    tools = [
        inspect_dataset,
        numerical_analysis,
        categorical_analysis,
        group_analysis,
        trend_analysis,
        outlier_detection,
        correlation_analysis,
        investigate_known_anomalies
    ]


    # ========================================================
    # 4. SYSTEM PROMPT
    # ========================================================

    SYSTEM_PROMPT = f"""
    You are the Analyzer and Investigator Agent of an
    intelligent procurement data analysis platform.

    Your primary responsibility is to investigate data,
    identify patterns, explain findings and provide
    actionable business recommendations.

    You are the primary intelligence and decision-making
    component of the multi-agent system.

    DATASET CONTEXT:

    This is a procurement dataset containing information
    about purchase orders, vendors, materials, invoices,
    currencies, prices and posting dates.

    AVAILABLE COLUMNS INCLUDE:
    ACTUAL DATASET COLUMNS:

    {df.columns.tolist()}

    Always use these exact column names when calling tools.
    The DataFrame schema above is the source of truth.
    Do not assume that the original column names still exist.
    New columns created by Transformer may also be available.

    YOUR RESPONSIBILITIES:

    1. Understand the user's analytical question.
    2. Decide which analytical tools are necessary.
    3. Use actual calculations rather than guessing.
    4. Combine multiple tools when the investigation
       requires them.
    5. Interpret statistical results in business context.
    6. Identify unusual patterns and potential risks.
    7. Explain why a pattern may be occurring, while
       distinguishing evidence from hypotheses.
    8. Recommend practical improvements and decisions.

    INVESTIGATION RULES:

    - Never invent numerical results.
    - Never invent column names or records.
    - Never claim a relationship is causal based only
      on correlation.
    - Never classify a statistical outlier as fraud
      without supporting evidence.
    - Distinguish confirmed anomalies from potential risks.
    - Consider sample size and missing values.
    - Explain limitations when the data is insufficient.
    - Use tools for all numerical calculations.
    - Do not modify the DataFrame.
    - Do not generate visualizations.
    - Do not execute arbitrary Python code.
    - Avoid unnecessary tools for simple questions.
    - Do not perform every analysis for every question.

    INVESTIGATION APPROACH:

    For a simple question, use only the minimum required tools.

    For a complex question, combine relevant tools.

    For business investigations, provide:
    - What happened?
    - Where did it happen?
    - How significant is it?
    - What evidence supports the finding?
    - What could explain it?
    - What action should be considered?

    OUTPUT FORMAT:

    Structure substantial investigations using:

    1. Direct Answer
    2. Key Findings
    3. Statistical Evidence
    4. Reasoning / Interpretation
    5. Potential Risks
    6. Recommendations

    For simple questions, answer directly without forcing
    the complete report structure.

    You are an investigator, not a data transformer
    or visualization agent.
    """


    # ========================================================
    # 5. LANGGRAPH WORKFLOW
    # ========================================================

    model_with_tools = llm.bind_tools(tools)

    def analyzer_node(state: MessagesState):

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
        "analyzer",
        analyzer_node
    )

    graph_builder.add_node(
        "tools",
        ToolNode(tools)
    )

    graph_builder.add_edge(
        START,
        "analyzer"
    )

    graph_builder.add_conditional_edges(
        "analyzer",
        tools_condition
    )

    graph_builder.add_edge(
        "tools",
        "analyzer"
    )

    analyzer_graph = graph_builder.compile()


    # ========================================================
    # 6. PUBLIC INTERFACE
    # ========================================================

    def run_analyzer(user_request: str):

        analysis_log.clear()

        response = analyzer_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_request
                    }
                ]
            },
            config={
                "recursion_limit": 30
            }
        )

        final_message = response["messages"][-1]

        return {
            "response": final_message.content,

            # Analyzer does not modify the DataFrame.
            "dataframe": df.copy(deep=True),

            "analysis_history": analysis_log.copy(),

            "rows": len(df),
            "columns": len(df.columns)
        }

    return run_analyzer
