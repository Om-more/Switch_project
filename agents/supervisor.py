
import os
from typing import TypedDict, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# --------------------------------------------------
# 1. SUPERVISOR PLAN
# --------------------------------------------------

AgentName = Literal["transformer", "analyzer", "visualizer"]


class SupervisorPlan(BaseModel):
    agents: list[AgentName] = Field(
        description=(
            "Workers required for the request, in execution order. "
            "Use only the necessary workers."
        )
    )

    transformer_task: str = Field(
        default="",
        description="Specific task for the Transformer."
    )

    analyzer_task: str = Field(
        default="",
        description="Specific task for the Analyzer."
    )

    visualizer_task: str = Field(
        default="",
        description="Specific task for the Visualizer."
    )

    explanation: str = Field(
        description="Brief explanation of why these workers are required."
    )


# --------------------------------------------------
# 2. SHARED LANGGRAPH STATE
# --------------------------------------------------

class CollaborationState(TypedDict):
    user_request: str

    dataframe: pd.DataFrame
    explicit_agent: Optional[str]
    agents: list[str]
    current_index: int

    transformer_task: str
    analyzer_task: str
    visualizer_task: str

    transformation_result: Optional[dict]
    analysis_result: Optional[dict]
    visualization_result: Optional[dict]

    execution_log: list[str]
    supervisor_explanation: str
    final_response: str


# --------------------------------------------------
# 3. SUPERVISOR FACTORY
# --------------------------------------------------

def build_supervisor_agent(
    initial_dataframe: pd.DataFrame,
    transformer_factory,
    analyzer_factory,
    visualizer_factory
):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ-key")
    )

    planner = llm.with_structured_output(SupervisorPlan)

    # --------------------------------------------------
    # 4. PLANNING NODE
    # --------------------------------------------------

    def plan_workflow(state: CollaborationState):
        # Explicit routing takes priority over automatic planning.
        explicit = state.get("explicit_agent")

        if explicit:

            task_fields = {
                "transformer": "transformer_task",
                "analyzer": "analyzer_task",
                "visualizer": "visualizer_task"
            }

            return {
                "agents": [explicit],
                "current_index": 0,

                "transformer_task": (
                    state["user_request"]
                    if explicit == "transformer" else ""
                ),

                "analyzer_task": (
                    state["user_request"]
                    if explicit == "analyzer" else ""
                ),

                "visualizer_task": (
                    state["user_request"]
                    if explicit == "visualizer" else ""
                ),

                "supervisor_explanation": (
                    f"User explicitly selected the {explicit} agent."
                ),

                "execution_log": [
                    f"Explicit routing: {explicit}"
                ]
            }
        prompt = f"""
You are the Supervisor of a procurement intelligence system.

You coordinate three independent specialist agents.

AGENT CAPABILITIES:

1. TRANSFORMER
- Cleans data.
- Removes duplicates.
- Handles missing values.
- Creates or modifies columns.
- Filters and transforms records.
- Changes the working DataFrame.

2. ANALYZER
- Performs statistical analysis.
- Detects anomalies and outliers.
- Investigates vendors and purchase patterns.
- Identifies trends and correlations.
- Is read-only.

3. VISUALIZER
- Creates chart and dashboard plans.
- Selects appropriate chart types.
- Identifies useful dashboard filters.
- Is read-only.

USER REQUEST:
{state["user_request"]}

DATASET:
Rows: {len(state["dataframe"])}
Columns: {list(state["dataframe"].columns)}

ROUTING RULES:

- Select only the workers needed.
- A simple cleaning request needs only Transformer.
- A statistical or anomaly investigation needs only Analyzer.
- A chart or dashboard request needs only Visualizer.
- If the user requests cleaning followed by investigation,
  execute Transformer before Analyzer.
- If the user requests analysis followed by a dashboard,
  execute Analyzer before Visualizer.
- If the user requests a complete workflow, use the
  appropriate sequence of workers.
- Analyzer must be allowed to work directly on the original
  DataFrame when transformation is not requested.
- Never select Transformer just because another agent is used.
- Do not invent tasks that the user did not request.
- Questions about counts, unique values, distinct values,
  totals, averages, maximums, minimums and percentages
  must be routed to Analyzer.

- Questions such as "How many vendors are there?"
  and "How many unique purchase orders exist?"
  must use Analyzer.

- Questions about vendor rankings, monthly comparisons,
  spending patterns, trends and unusual expenditure
  must use Analyzer.

- Requests to modify, clean, rename, filter or create
  DataFrame columns must use Transformer.

- Requests for charts, graphs, plots or dashboards
  must use Visualizer.

EXAMPLES:

"How many vendors are there?"
-> analyzer

"Which vendor has the highest expenditure?"
-> analyzer

"Remove duplicate records."
-> transformer

"Show monthly expenditure by vendor."
-> visualizer

"Clean the data and find unusual vendor spending."
-> transformer -> analyzer

"Find unusual vendor spending and visualize it."
-> analyzer -> visualizer

Return the required agents in execution order and give each
worker a clear, specific task.
"""

        plan = planner.invoke(prompt)

        agents = list(dict.fromkeys(plan.agents))

        if not agents:
            raise ValueError(
                "Supervisor could not identify a required worker."
            )

        return {
            "agents": agents,
            "current_index": 0,
            "transformer_task": plan.transformer_task,
            "analyzer_task": plan.analyzer_task,
            "visualizer_task": plan.visualizer_task,
            "supervisor_explanation": plan.explanation,
            "execution_log": [
                "Supervisor selected: " + " -> ".join(agents)
            ]
        }

    # --------------------------------------------------
    # 5. WORKER INVOCATION HELPER
    # --------------------------------------------------

    def invoke_worker(worker, task):

        # LangGraph agents generally accept a messages dictionary.
        if hasattr(worker, "invoke"):
            return worker.invoke({
                "messages": [HumanMessage(content=task)]
            })

        # Supports callable worker wrappers as well.
        return worker(task)

    def get_result_text(result):

        if isinstance(result, dict):
            if "response" in result:
                return str(result["response"])

            messages = result.get("messages", [])
            if messages:
                return str(messages[-1].content)

        if hasattr(result, "content"):
            return str(result.content)

        return str(result)

    # --------------------------------------------------
    # 6. TRANSFORMER NODE
    # --------------------------------------------------

    def run_transformer(state: CollaborationState):

        worker = transformer_factory(state["dataframe"])

        result = invoke_worker(
            worker,
            state["transformer_task"]
        )

        updated_df = (
            result.get("dataframe")
            if isinstance(result, dict)
            else None
        )

        if isinstance(updated_df, pd.DataFrame):
            current_df = updated_df
        else:
            current_df = state["dataframe"]

        return {
            "dataframe": current_df,
            "transformation_result": result,
            "current_index": state["current_index"] + 1,
            "execution_log": state["execution_log"] + [
                "Transformer completed."
            ]
        }

    # --------------------------------------------------
    # 7. ANALYZER NODE
    # --------------------------------------------------

    def run_analyzer(state: CollaborationState):

        # Always build Analyzer using the latest available DataFrame.
        worker = analyzer_factory(state["dataframe"])

        result = invoke_worker(
            worker,
            state["analyzer_task"]
        )

        return {
            "analysis_result": result,
            "current_index": state["current_index"] + 1,
            "execution_log": state["execution_log"] + [
                "Analyzer completed."
            ]
        }

    # --------------------------------------------------
    # 8. VISUALIZER NODE
    # --------------------------------------------------

    def run_visualizer(state: CollaborationState):

        worker = visualizer_factory(state["dataframe"])

        result = invoke_worker(
            worker,
            state["visualizer_task"]
        )

        return {
            "visualization_result": result,
            "current_index": state["current_index"] + 1,
            "execution_log": state["execution_log"] + [
                "Visualizer completed."
            ]
        }

    # --------------------------------------------------
    # 9. CONDITIONAL ROUTING
    # --------------------------------------------------

    def route_to_worker(state: CollaborationState):

        agents = state["agents"]
        index = state["current_index"]

        if index >= len(agents):
            return "finish"

        return agents[index]

    # --------------------------------------------------
    # 10. FINAL RESPONSE NODE
    # --------------------------------------------------

    def create_final_response(state: CollaborationState):

        sections = []

        if state["transformation_result"] is not None:
            sections.append(
                "TRANSFORMATION:\n" +
                get_result_text(state["transformation_result"])
            )

        if state["analysis_result"] is not None:
            sections.append(
                "ANALYSIS:\n" +
                get_result_text(state["analysis_result"])
            )

        if state["visualization_result"] is not None:
            sections.append(
                "VISUALIZATION:\n" +
                get_result_text(state["visualization_result"])
            )

        response = "\n\n".join(sections)

        return {
            "final_response": (
                "Workflow completed.\n\n"
                + state["supervisor_explanation"]
                + "\n\n"
                + response
            )
        }

    # --------------------------------------------------
    # 11. BUILD GRAPH
    # --------------------------------------------------

    graph = StateGraph(CollaborationState)

    graph.add_node("planner", plan_workflow)
    graph.add_node("transformer", run_transformer)
    graph.add_node("analyzer", run_analyzer)
    graph.add_node("visualizer", run_visualizer)
    graph.add_node("finish", create_final_response)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_to_worker,
        {
            "transformer": "transformer",
            "analyzer": "analyzer",
            "visualizer": "visualizer",
            "finish": "finish"
        }
    )

    for worker_name in ["transformer", "analyzer", "visualizer"]:
        graph.add_conditional_edges(
            worker_name,
            route_to_worker,
            {
                "transformer": "transformer",
                "analyzer": "analyzer",
                "visualizer": "visualizer",
                "finish": "finish"
            }
        )

    graph.add_edge("finish", END)

    return graph.compile()


# --------------------------------------------------
# 12. USER-FACING RUN FUNCTION
# --------------------------------------------------

def run_collaboration(
    supervisor,
    dataframe,
    user_request,
    explicit_agent=None
):

    allowed_agents = [
        "transformer",
        "analyzer",
        "visualizer"
    ]

    if explicit_agent is not None:
        if explicit_agent not in allowed_agents:
            raise ValueError(
                f"Invalid agent: {explicit_agent}"
            )

    initial_state = {
        "user_request": user_request,
        "dataframe": dataframe.copy(deep=True),

        "explicit_agent": explicit_agent,

        "agents": [],
        "current_index": 0,

        "transformer_task": "",
        "analyzer_task": "",
        "visualizer_task": "",

        "transformation_result": None,
        "analysis_result": None,
        "visualization_result": None,

        "execution_log": [],
        "supervisor_explanation": "",
        "final_response": ""
    }

    return supervisor.invoke(initial_state)