
import operator
import json
import pandas as pd

from typing import Literal
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# =========================================================
# 1. LLM CONFIGURATION
# =========================================================

import os
from dotenv import load_dotenv

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


# =========================================================
# 2. TRANSFORMER AGENT FACTORY
# =========================================================

def build_transformer_agent(dataframe):

    # Keep the original DataFrame unchanged.
    original_df = dataframe.copy(deep=True)

    # This is the working DataFrame.
    working_df = dataframe.copy(deep=True)

    transformation_log = []

    # -----------------------------------------------------
    # TOOL 1: INSPECT DATA
    # -----------------------------------------------------

    @tool
    def inspect_data() -> str:
        """
        Inspect the current DataFrame.

        Returns column names, data types, row count,
        missing values, duplicate count and sample records.

        Always use this before performing unfamiliar
        transformations.
        """

        report = {
            "total_rows": len(working_df),
            "total_columns": len(working_df.columns),
            "columns": working_df.columns.tolist(),
            "data_types": {
                col: str(dtype)
                for col, dtype in working_df.dtypes.items()
            },
            "missing_values": {
                col: int(count)
                for col, count in working_df.isnull().sum().items()
            },
            "duplicate_rows": int(working_df.duplicated().sum()),
            "sample_data": working_df.head(5).to_dict(
                orient="records"
            )
        }

        return json.dumps(report, default=str, indent=2)


    # -----------------------------------------------------
    # TOOL 2: REMOVE DUPLICATES
    # -----------------------------------------------------

    @tool
    def remove_duplicates(
        subset: list[str] | None = None,
        keep: Literal["first", "last", "none"] = "first"
    ) -> str:
        """
        Remove duplicate rows.

        subset: Optional list of columns used to identify duplicates.
        keep: Keep first, last, or none of the duplicate records.
        """

        nonlocal working_df

        if subset:
            missing = [c for c in subset if c not in working_df.columns]
            if missing:
                return f"Columns not found: {missing}"

        before = len(working_df)

        keep_value = None if keep == "none" else keep

        working_df = working_df.drop_duplicates(
            subset=subset,
            keep=keep_value
        ).copy()

        removed = before - len(working_df)

        transformation_log.append(
            f"Removed {removed} duplicate rows."
        )

        return f"Removed {removed} duplicate rows. Remaining: {len(working_df)}"


    # -----------------------------------------------------
    # TOOL 3: REMOVE COLUMNS
    # -----------------------------------------------------

    @tool
    def drop_columns(columns: list[str]) -> str:
        """
        Remove specified columns from the DataFrame.
        """

        nonlocal working_df

        missing = [c for c in columns if c not in working_df.columns]

        if missing:
            return f"Columns not found: {missing}"

        working_df = working_df.drop(columns=columns).copy()

        transformation_log.append(
            f"Dropped columns: {columns}"
        )

        return f"Successfully removed columns: {columns}"


    # -----------------------------------------------------
    # TOOL 4: RENAME COLUMNS
    # -----------------------------------------------------

    @tool
    def rename_columns(mapping: dict[str, str]) -> str:
        """
        Rename DataFrame columns.

        mapping example:
        {"Vendor Code": "Supplier ID"}
        """

        nonlocal working_df

        missing = [
            col for col in mapping
            if col not in working_df.columns
        ]

        if missing:
            return f"Columns not found: {missing}"

        new_names = [
            mapping.get(col, col)
            for col in working_df.columns
        ]

        if len(new_names) != len(set(new_names)):
            return "Rename rejected: duplicate column names would be created."

        working_df = working_df.rename(
            columns=mapping
        ).copy()

        transformation_log.append(
            f"Renamed columns: {mapping}"
        )

        return f"Successfully renamed columns: {mapping}"


    # -----------------------------------------------------
    # TOOL 5: HANDLE MISSING VALUES
    # -----------------------------------------------------

    @tool
    def fill_missing_values(
        column: str,
        strategy: Literal["mean", "median", "mode", "constant"],
        value: str | None = None
    ) -> str:
        """
        Fill missing values in a specified column.

        strategy:
        mean, median, mode or constant.

        value:
        Required only when strategy is constant.
        """

        nonlocal working_df

        if column not in working_df.columns:
            return f"Column not found: {column}"

        missing_before = int(working_df[column].isnull().sum())

        if missing_before == 0:
            return f"No missing values found in {column}."

        series = working_df[column]

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                return "Mean can only be used on numeric columns."

            fill_value = series.mean()

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                return "Median can only be used on numeric columns."

            fill_value = series.median()

        elif strategy == "mode":
            modes = series.mode(dropna=True)

            if modes.empty:
                return "Cannot calculate mode: column has no valid values."

            fill_value = modes.iloc[0]

        elif strategy == "constant":
            if value is None:
                return "Please provide a constant value."

            fill_value = value

            if pd.api.types.is_numeric_dtype(series):
                try:
                    fill_value = pd.to_numeric(value)
                except (ValueError, TypeError):
                    return "Provided value is not numeric."

        else:
            return "Unsupported filling strategy."

        working_df[column] = series.fillna(fill_value)

        transformation_log.append(
            f"Filled {missing_before} missing values in {column} using {strategy}."
        )

        return (
            f"Filled {missing_before} missing values "
            f"in {column} using {strategy}."
        )


    # -----------------------------------------------------
    # TOOL 6: REPLACE VALUES
    # -----------------------------------------------------

    @tool
    def replace_values(
        column: str,
        old_value: str,
        new_value: str
    ) -> str:
        """
        Replace a specific value with another value
        in a specified column.
        """

        nonlocal working_df

        if column not in working_df.columns:
            return f"Column not found: {column}"

        old_value_converted = old_value
        new_value_converted = new_value

        if pd.api.types.is_numeric_dtype(working_df[column]):
            try:
                old_value_converted = pd.to_numeric(old_value)
                new_value_converted = pd.to_numeric(new_value)
            except (ValueError, TypeError):
                return "Replacement values must be numeric for this column."

        matches = int(
            (working_df[column] == old_value_converted).sum()
        )

        if matches == 0:
            return f"No matching values found for {old_value}."

        working_df[column] = working_df[column].replace(
            old_value_converted,
            new_value_converted
        )

        transformation_log.append(
            f"Replaced {old_value} with {new_value} in {column}."
        )

        return f"Replaced {matches} occurrences in {column}."


    # -----------------------------------------------------
    # TOOL 7: FILTER ROWS
    # -----------------------------------------------------

    @tool
    def filter_rows(
        column: str,
        condition: Literal[
            "equals",
            "not_equals",
            "greater_than",
            "less_than",
            "greater_equal",
            "less_equal",
            "contains"
        ],
        value: str
    ) -> str:
        """
        Filter rows using a specified condition.

        Examples:
        Currency equals INR
        Line Amount greater_than 10000
        Vendor Name contains Supplier

        Filtering changes the working DataFrame.
        """

        nonlocal working_df

        if column not in working_df.columns:
            return f"Column not found: {column}"

        series = working_df[column]

        if condition == "contains":
            mask = series.astype(str).str.contains(
                value,
                case=False,
                na=False,
                regex=False
            )

        else:
            if pd.api.types.is_numeric_dtype(series):
                try:
                    compare_value = pd.to_numeric(value)
                except (ValueError, TypeError):
                    return "Value must be numeric for this column."
            else:
                compare_value = value

            operations = {
                "equals": operator.eq,
                "not_equals": operator.ne,
                "greater_than": operator.gt,
                "less_than": operator.lt,
                "greater_equal": operator.ge,
                "less_equal": operator.le
            }

            mask = operations[condition](
                series,
                compare_value
            )

        before = len(working_df)

        working_df = working_df.loc[mask].copy()

        transformation_log.append(
            f"Filtered {column} using {condition} {value}."
        )

        return (
            f"Filtering successful. "
            f"Rows before: {before}, "
            f"Rows after: {len(working_df)}"
        )


    # -----------------------------------------------------
    # TOOL 8: CREATE CALCULATED COLUMN
    # -----------------------------------------------------

    @tool
    def create_numeric_column(
        new_column: str,
        column_a: str,
        operation: Literal["add", "subtract", "multiply", "divide"],
        column_b: str | None = None,
        constant: float | None = None
    ) -> str:
        """
        Create a new numeric column using two existing
        numeric columns or one existing column and a constant.

        Provide either column_b OR constant, not both.

        Example:
        new_column = "Calculated Amount"
        column_a = "Unit Price"
        operation = "multiply"
        column_b = "Currency Factor"
        """

        nonlocal working_df

        if column_a not in working_df.columns:
            return f"Column not found: {column_a}"

        if (column_b is None) == (constant is None):
            return "Provide exactly one of column_b or constant."

        if column_b is not None:
            if column_b not in working_df.columns:
                return f"Column not found: {column_b}"

            if not pd.api.types.is_numeric_dtype(working_df[column_b]):
                return f"{column_b} is not numeric."

            right = working_df[column_b]

        else:
            right = constant

        if not pd.api.types.is_numeric_dtype(working_df[column_a]):
            return f"{column_a} is not numeric."

        left = working_df[column_a]

        operations = {
            "add": operator.add,
            "subtract": operator.sub,
            "multiply": operator.mul,
            "divide": operator.truediv
        }

        if operation == "divide":
            if isinstance(right, pd.Series):
                if (right == 0).any():
                    return "Division rejected: denominator contains zero."
            elif right == 0:
                return "Division rejected: denominator is zero."

        try:
            result = operations[operation](left, right)
        except Exception as e:
            return f"Calculation failed: {str(e)}"

        if new_column in working_df.columns:
            return f"Column {new_column} already exists. Rename it first."

        working_df[new_column] = result

        transformation_log.append(
            f"Created {new_column} using {column_a} {operation} {column_b or constant}."
        )

        return f"Successfully created calculated column: {new_column}"


    # -----------------------------------------------------
    # TOOL 9: VIEW TRANSFORMATION HISTORY
    # -----------------------------------------------------

    @tool
    def get_transformation_history() -> str:
        """
        Return the list of transformations performed so far.
        """

        if not transformation_log:
            return "No transformations have been performed."

        return "\n".join(
            f"{i + 1}. {entry}"
            for i, entry in enumerate(transformation_log)
        )


    # -----------------------------------------------------
    # 3. REGISTER TOOLS
    # -----------------------------------------------------

    tools = [
        inspect_data,
        remove_duplicates,
        drop_columns,
        rename_columns,
        fill_missing_values,
        replace_values,
        filter_rows,
        create_numeric_column,
        get_transformation_history
    ]


    # -----------------------------------------------------
    # 4. AGENT SYSTEM INSTRUCTIONS
    # -----------------------------------------------------

    SYSTEM_PROMPT = """
    You are the Transformer Agent of a multi-agent
    data intelligence platform.

    Your responsibility is to understand user requests
    and perform data transformations using the provided tools.

    YOUR RESPONSIBILITIES:

    1. Understand the requested transformation.
    2. Inspect the DataFrame when necessary.
    3. Identify the relevant columns.
    4. Select the appropriate tools.
    5. Execute transformations in the correct sequence.
    6. Validate the result.
    7. Explain what changes were performed.

    IMPORTANT RULES:

    - Never invent column names.
    - Never invent data values.
    - Never generate or execute arbitrary Python code.
    - Use only the available tools for transformations.
    - Do not perform unrelated transformations.
    - If the request is ambiguous, ask for clarification.
    - If an operation cannot be safely performed, explain why.
    - Do not claim that a transformation succeeded unless
      the corresponding tool confirms it.
    - Inspect data before unfamiliar operations.
    - Preserve the original DataFrame.
    - Do not write changes back to Supabase.
    - Filtering is also a transformation and changes the
      working dataset. Explain this to the user.

    You can execute multiple tools when a request requires
    multiple operations.

    After completing the task, summarize:
    - What was changed.
    - Why it was changed.
    - Number of affected rows or columns, when available.
    - Current DataFrame dimensions.

    You are a data transformation specialist, not an
    analytics or visualization agent.
    """


    # -----------------------------------------------------
    # 5. LANGGRAPH WORKFLOW
    # -----------------------------------------------------

    model_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):

        response = model_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"]
            ]
        )

        return {"messages": [response]}


    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("transformer", agent_node)

    graph_builder.add_node(
        "tools",
        ToolNode(tools)
    )

    graph_builder.add_edge(START, "transformer")

    graph_builder.add_conditional_edges(
        "transformer",
        tools_condition
    )

    graph_builder.add_edge("tools", "transformer")

    transformer_graph = graph_builder.compile()


    # -----------------------------------------------------
    # 6. PUBLIC INTERFACE
    # -----------------------------------------------------

    def run_transformer(user_request: str):

        response = transformer_graph.invoke(
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
            "dataframe": working_df.copy(deep=True),
            "original_dataframe": original_df.copy(deep=True),
            "transformation_history": transformation_log.copy(),
            "rows": len(working_df),
            "columns": len(working_df.columns)
        }

    return run_transformer
