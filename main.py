
import re
from pathlib import Path

import pandas as pd
import subprocess
import sys
from supabase_pull import fetch_all

from agents.transformation import build_transformer_agent
from agents.analyzer import build_analyzer_agent
from agents.visualizer import build_visualizer_agent

from agents.supervisor import (
    build_supervisor_agent,
    run_collaboration
)

def launch_visualizer():
    print("\nLaunching Streamlit Visualizer...")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "visualizer_app.py"
        ]
    )

    print("Streamlit is starting.")
    print("Enter your visualization requests in the browser.")
    
def main():

    # --------------------------------------------------
    # 1. FETCH DATA
    # --------------------------------------------------

    print("\nFetching data from Supabase...")

    df = fetch_all()

    if df.empty:
        print("No records found.")
        return

    print(f"Loaded {len(df)} records.")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------
    # 2. LOAD KNOWN ANOMALIES
    # --------------------------------------------------

    anomaly_file = Path("Known_anomalies.xlsx")

    known_anomalies = (
        pd.read_excel(anomaly_file)
        if anomaly_file.exists()
        else None
    )

    # --------------------------------------------------
    # 3. BUILD SUPERVISOR
    # --------------------------------------------------

    supervisor = build_supervisor_agent(

        initial_dataframe=df,

        transformer_factory=build_transformer_agent,

        analyzer_factory=lambda data: build_analyzer_agent(
            data,
            known_anomalies=known_anomalies
        ),

        visualizer_factory=build_visualizer_agent
    )

    print("\nProcurement Intelligence System is ready!")
    print("Model: Groq GPT-OSS-120B")

    print("\nAvailable commands:")
    print("@transformer <request>")
    print("@analyzer <request>")
    print("@visualizer <request>")
    print("Or simply type your question for automatic routing.")

    print("\nType 'exit' to quit.")

    # --------------------------------------------------
    # 4. INTERACTIVE SUPERVISOR LOOP
    # --------------------------------------------------

    while True:

        raw_input_text = input("\nYou: ").strip()

        if raw_input_text.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not raw_input_text:
            continue

        # Launch Streamlit visualizer when requested
        if raw_input_text.lower() == "@visualizer":
            print("\nLaunching Streamlit Visualizer...")
            launch_visualizer()
            continue

        # --------------------------------------------------
        # 5. DETECT EXPLICIT AGENT SELECTION
        # --------------------------------------------------

        match = re.match(
            r"^@(transformer|analyzer|visualizer)\b\s*(.*)$",
            raw_input_text,
            flags=re.IGNORECASE
        )

        explicit_agent = None

        if match:

            explicit_agent = match.group(1).lower()
            user_request = match.group(2).strip()

            if not user_request:
                print("Please provide a request after the agent name.")
                continue

        else:

            user_request = raw_input_text

        # --------------------------------------------------
        # 6. SEND REQUEST THROUGH SUPERVISOR
        # --------------------------------------------------

        try:

            result = run_collaboration(
                supervisor=supervisor,
                dataframe=df,
                user_request=user_request,
                explicit_agent=explicit_agent
            )

            df = result["dataframe"]

            print("\nExecution Log:")

            for step in result["execution_log"]:
                print("-", step)

            print("\nSupervisor Response:")
            print(result["final_response"])

            visual_result = result.get("visualization_result")

            if visual_result is not None:
                print("\nVisualization Plan:")
                print(
                    visual_result.get(
                        "visualization_plan",
                        {}
                    )
                )

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
