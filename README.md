<div align="center">
# ProcureLens
### AI-Powered Procurement Intelligence

**Turn raw procurement records into explainable insights, anomaly investigations, and interactive visualizations.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq-F55036)](https://groq.com/)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

[Repository](https://github.com/Om-more/Switch_project)

</div>

---
ProcureLens is a multi-agent procurement analytics prototype. It uses LangGraph to coordinate specialist agents that transform procurement data, investigate spending patterns, and generate interactive visualizations.

The current project connects to a Supabase PostgreSQL table, uses Groq's `openai/gpt-oss-120b` model for agent reasoning, and provides terminal and Streamlit interfaces.

> **Project status:** The repository currently contains the three specialist agents, a LangGraph supervisor, Supabase data retrieval, and a Streamlit visualization interface. Switch and Discord/Microsoft Teams integration are planned next steps; they are not included in the current code described here.

---

## What ProcureLens does

Procurement records can contain inconsistent supplier names, unusual prices, invoice differences, currency conversion issues, and spending patterns that are difficult to inspect manually.

ProcureLens helps a procurement or finance team explore these records through three specialist agents:

| Agent | Responsibility |
|---|---|
| **Transformer Agent** | Inspects and prepares procurement data, applies transformations, and supports data-quality operations. |
| **Analyzer / Investigator Agent** | Performs procurement analysis, investigates anomalies and spending patterns, and explains findings. |
| **Visualizer Agent** | Understands visualization requests and prepares Plotly charts and visualization plans. |

A LangGraph **Supervisor** interprets the user's request and selects the required agent or sequence of agents. Users can also explicitly call an agent using terminal commands.

## Architecture

```text
                         User request
                              |
                              v
                    LangGraph Supervisor
                              |
                 Selects required specialist(s)
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
       Transformer        Analyzer          Visualizer
          Agent             Agent              Agent
            |                 |                 |
            +-----------------+-----------------+
                              |
                              v
                       Final response
                              |
                 +------------+------------+
                 |                         |
                 v                         v
           Terminal CLI             Streamlit UI

                 Supabase PostgreSQL
                         ^
                         |
                supabase_pull.py
```

The agents use LangChain tools and LangGraph workflows. Groq provides the language model used for planning, tool selection, and natural-language explanations. Pandas and NumPy handle data operations and calculations; Plotly generates visualizations.

## Technology stack

- **Python 3.11**
- **LangGraph** — agent workflows and supervisor orchestration
- **LangChain** — LLM and tool integration
- **Groq** — model inference
- **GPT-OSS 120B** — `openai/gpt-oss-120b`
- **Supabase** — PostgreSQL procurement data
- **Pandas / NumPy** — data processing and analytics
- **Plotly** — interactive charts
- **Streamlit** — visualization interface
- **python-dotenv** — environment configuration

---

## Repository structure

```text
Switch_project/
├── agents/
│   ├── transformation.py     # Transformer Agent
│   ├── analyzer.py           # Analyzer / Investigator Agent
│   ├── visualizer.py         # Visualizer Agent and Plotly helpers
│   └── supervisor.py         # LangGraph Supervisor and collaboration runner
│
├── .env                      # Local credentials (do not commit)
├── .gitignore
├── Procure_data.xlsx         # Local procurement data/reference workbook
├── Known_anomalies.xlsx      # Known anomaly reference/evaluation workbook
├── supabase_pull.py          # Supabase connection and paginated data retrieval
├── main.py                   # Terminal-based interactive application
├── test-analyzer.py          # Interactive Analyzer test script
├── visualizer_app.py         # Streamlit visualization application
├── app.py                    # Streamlit procurement dashboard
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Om-more/Switch_project.git
cd Switch_project
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages used by the current application:

```bash
python -m pip install --upgrade pip
pip install pandas numpy python-dotenv supabase langchain langchain-core langchain-groq langgraph pydantic plotly streamlit openpyxl
```

If the network is slow, increase pip's timeout:

```bash
python -m pip install pandas numpy python-dotenv supabase langchain langchain-core langchain-groq langgraph pydantic plotly streamlit openpyxl --timeout 120 --retries 10
```

For reproducible installs, create and maintain a pinned `requirements.txt` after verifying the working environment:

```bash
pip freeze > requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root. The current source code reads the following exact variable names:

```env
GROQ-key=your_groq_api_key
project-url=https://your-project-id.supabase.co
supabase-api-key=your_supabase_key
```

**Important:**
- Do not commit `.env` or real credentials.
- `.gitignore` already excludes `.env` and `.venv/`.
- Use a Supabase key with only the permissions required by the application. Keep privileged/service-role keys strictly on the server side.
- The current code uses the names above, including their capitalization and hyphens. If you rename them, update the corresponding `os.getenv(...)` calls in the Python files as well.

### 5. Configure Supabase data

The current Supabase helper reads from the table:

```text
Procure_data
```

The helper in `supabase_pull.py` uses pagination with a default batch size of 1,000 rows and continues until all available records are retrieved.

Check that:
- The table exists in your Supabase project.
- The configured key can read the table.
- The table's column names match the fields expected by the agents.
- Row-level security policies permit the intended read operation.

---

## Run ProcureLens

### Option A — Interactive terminal application

From the repository root, with the virtual environment activated:

```bash
python main.py
```

The application fetches procurement records from Supabase, initializes the specialist agents and Supervisor, and opens an interactive prompt.

Example commands:

```text
@transformer clean and inspect the procurement data
@analyzer identify suppliers with unusually high spending
@visualizer show monthly procurement spending
```

You can also enter a natural-language request without specifying an agent. The Supervisor will plan which specialist agent(s) should handle it.

To exit:

```text
exit
```

### Option B — Run the Analyzer independently

```bash
python test-analyzer.py
```

This script fetches procurement data, loads the local `Known_anomalies.xlsx` reference file when available, initializes the Analyzer, and accepts analysis questions in the terminal.

### Option C — Launch the Streamlit visualizer

```bash
streamlit run visualizer_app.py
```

The terminal application can also launch this interface when the user enters:

```text
@visualizer
```

### Option D — Launch the alternate Streamlit app

```bash
streamlit run app.py
```

This app fetches data from Supabase, accepts a visualization request, and renders the generated Plotly charts with filters.

---

## Example procurement questions

Try requests such as:

**Data transformation**
- Inspect missing values and data types.
- Identify duplicate procurement records.
- Normalize supplier names.
- Check currency factors and invoice differences.

**Procurement investigation**
- Which suppliers account for the highest expenditure?
- Show monthly procurement spending trends.
- Compare prices for the same material across suppliers.
- Identify potential invoice amount discrepancies.
- Which procurement categories contribute most to spending?
- Identify unusual currency conversion factors.

**Visualization**
- Show monthly procurement expenditure.
- Compare the top 10 suppliers by total spend.
- Visualize category-wise expenditure.
- Show state-wise procurement spending.
- Compare supplier prices for a selected material.

The exact availability of an analysis depends on the columns and records present in the connected dataset.

---

## Data and evaluation notes

The repository includes:

- `Procure_data.xlsx` — local procurement data/reference workbook.
- `Known_anomalies.xlsx` — known anomaly reference workbook used by the current Analyzer test/application setup.

The live application retrieves its working procurement records from Supabase. The local workbooks are not automatically synchronized with Supabase.

**Evaluation caution:** For a blind anomaly-detection evaluation, do not provide known anomaly labels to the agent while it is making predictions. Keep the ground-truth workbook separate and compare agent findings against it after execution. The current `main.py` and `test-analyzer.py` load the known-anomaly workbook and pass it to the Analyzer; change that behavior for a strict blind evaluation.

The available procurement fields do not necessarily include purchase quantity, contract terms, payment status, or supplier performance history. Do not infer or report findings that require data not present in the source.

---

## Current limitations and next steps

The current repository is a local prototype. Planned improvements include:

1. Add automated tests for all three agents and Supervisor routing.
2. Separate reusable database, analytical, and visualization utilities into dedicated modules.
3. Add robust logging, validation, and error handling.
4. Add a persistent dashboard-serving option if visualizations need to be opened from another application.
5. Integrate ProcureLens with **Switch** using the official Switch agent protocol.
6. Connect a collaboration platform such as **Discord** or **Microsoft Teams** through Switch.
7. Add human review and approval flows for recommended procurement actions.

Switch and Discord/Teams integration are not yet implemented in the current repository. The intended future flow is:

```text
Discord / Microsoft Teams
          |
          v
     Switch Rooms
          |
          v
  ProcureLens Switch Adapter
          |
          v
  Existing LangGraph Supervisor
          |
          v
 Transformer / Analyzer / Visualizer
          |
          v
   Supabase procurement data
```

---

## Security

- Keep `.env` out of version control.
- Never paste API keys into source code, README files, screenshots, or chat messages.
- Rotate any credential that has accidentally been committed or exposed.
- Avoid logging complete procurement records or sensitive supplier information.
- Use least-privilege database credentials and appropriate Supabase row-level security.
- Treat agent-generated anomaly explanations as investigation leads, not proof of fraud or misconduct.

---

## Project

**ProcureLens — Procurement Intelligence & Investigation using Collaborative AI Agents**

Repository: https://github.com/Om-more/Switch_project
