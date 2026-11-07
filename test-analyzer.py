
import pandas as pd

from supabase_pull import fetch_all
from agents.analyzer import build_analyzer_agent


def main():

    # 1. Fetch procurement data
    print("\nFetching procurement data...")

    df = fetch_all()

    print("Records:", len(df))

    # 2. Load known anomaly reference
    try:
        known_anomalies = pd.read_excel(
            "Known_anomalies.xlsx"
        )

        print(
            "Known anomalies loaded:",
            len(known_anomalies)
        )

    except FileNotFoundError:
        known_anomalies = None

        print("Known anomaly file not found.")

    # 3. Initialize Analyzer
    analyzer = build_analyzer_agent(
        dataframe=df,
        known_anomalies=known_anomalies
    )

    print("\nAnalyzer Agent is ready!")
    print("Model: Groq GPT-OSS-120B")
    print("Type 'exit' to stop.")

    # 4. Interactive session
    while True:

        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        try:

            result = analyzer(query)

            print("\n" + "=" * 60)
            print("ANALYZER / INVESTIGATOR")
            print("=" * 60)

            print(result["response"])

            print("\nInvestigation History:")

            for item in result["analysis_history"]:
                print("-", item)

            print("\nDataset Information:")
            print("Rows:", result["rows"])
            print("Columns:", result["columns"])

        except Exception as e:
            print("Analysis error:", str(e))


if __name__ == "__main__":
    main()
