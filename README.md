# Agentic LLM Benchmarking Workbench

Generate answers with selected OpenAI models, assess them with specialist evaluator agents for faithfulness, toxicity, bias, and reasoning quality, aggregate the evidence with a separate judge agent, and inspect MLflow runs and OpenTelemetry traces.

The workbench separates three roles:

```text
Candidate OpenAI configurations
        |
        | generate answers
        v
Specialist evaluator agents
  - bias
  - toxicity
  - faithfulness
  - reasoning
        |
        v
Separate final judge
        |
        v
Results dashboard + MLflow + OpenTelemetry + JSON evidence bundle
```

**Developed by Mayukh Das**  
TU Braunschweig  
mayukh@ifis.cs.tu-bs.de

---

## How to use the workbench

1. **Enter a question or task** — provide a question or instruction and optionally add reference context for faithfulness evaluation.
2. **Select candidate models** — choose up to three curated OpenAI model configurations. The candidate models generate answers, while the evaluation preset in the left sidebar controls the separate evaluator and judge models.
3. **Run and inspect** — compare model quality, safety, faithfulness, reasoning, latency, estimated cost, MLflow runs, and OpenTelemetry traces.

The main interface intentionally hides raw candidate IDs and repeated model-detail text. Exact model IDs, reasoning settings, evaluator settings, and judge settings remain available in the result metadata, MLflow records, and downloadable evidence reports.

---

## Why this project exists

Many LLM demonstrations stop at a chatbot or a simple prompt chain. This project instead demonstrates several skills that are relevant to LLM evaluation and AI engineering:

- multi-step agent orchestration;
- separation of candidate, evaluator, and judge models;
- rubric-based structured evaluation;
- blind model assessment and randomized execution order;
- experiment tracking with MLflow;
- distributed-style tracing with OpenTelemetry;
- token, latency, and estimated-cost monitoring;
- reproducible JSON and CSV evidence exports;
- an interactive Streamlit interface suitable for a portfolio demonstration;
- a prominent benchmark action, large bold custom-styled result tabs, and compact summary cards that keep model names readable without truncation.

The system is an **evaluation workbench**, not a claim that LLM-as-a-judge scores are objective ground truth. Its reports preserve evidence, uncertainty, configuration, and trace metadata so that results can be inspected critically.

---

## Main features

### Candidate answer generation

The user provides only:

- a question or task;
- optional reference context.

The selected candidate configurations generate their own answers. The user does not need to paste model responses manually.

### Specialized evaluator agents

Each answer is assessed by four separate rubric agents:

1. **Bias agent** — demographic stereotypes, unfair generalizations, asymmetric standards, and demeaning associations.
2. **Toxicity agent** — abusive language, hate, threats, harassment, dehumanization, and violent hostility.
3. **Faithfulness agent** — unsupported claims, contradictions, fabricated details, and unjustified certainty relative to the supplied context.
4. **Reasoning agent** — logical consistency, relevance, causal validity, support for conclusions, and treatment of uncertainty.

A separate **judge agent** aggregates these assessments into an overall verdict and score.

### Blind evaluation

Evaluator prompts receive an anonymized candidate answer, not the candidate model name. Candidate execution order is randomized for every benchmark. The final report restores the real model configuration for reproducibility and presentation.

### Two clear presets

- **Strict** — default; GPT-5.4 mini with medium reasoning for evaluator agents and high reasoning for the final judge.
- **Balanced** — GPT-4o mini for evaluator agents and the judge, providing a cheaper and faster demonstration.

There are no custom provider placeholders or offline heuristic scores. If a key or model is unavailable, the application reports the failure explicitly.

### Eight curated candidate configurations

The dropdown contains eight low-cost OpenAI configurations:

1. GPT-4o mini
2. GPT-4.1 mini
3. GPT-5 nano — low reasoning
4. GPT-5 mini — low reasoning
5. GPT-5.4 nano — low reasoning
6. GPT-5.4 nano — high reasoning
7. GPT-5.4 mini — low reasoning
8. GPT-5.4 mini — medium reasoning

The three default candidates are:

- GPT-4o mini;
- GPT-5 mini — low reasoning;
- GPT-5.4 nano — low reasoning.

Model IDs, reasoning effort, descriptions, and price assumptions are stored in `data/models.json`, so the catalog can be updated without rewriting application logic.

> Pricing changes over time. The values in `data/models.json` are estimates used for demonstration and should be checked against the current OpenAI model pages before publishing benchmark costs.

---

## Result visualizations

Results are organized into six prominent tabs: **Overview**, **Model Comparison**, **Agent Findings**, **MLflow**, **OpenTelemetry Traces**, and **Downloads**. Custom scoped CSS makes the complete tab labels larger and bolder, while horizontal scrolling preserves them on narrow screens. The Overview uses custom summary cards so model families appear at a medium, readable size and reasoning configurations appear as secondary text instead of being truncated.

The Streamlit dashboard includes:

- headline cards for best overall model, fastest model, best-value model, and estimated API cost;
- a metric heatmap covering bias safety, toxicity safety, faithfulness, reasoning, and overall score;
- a quality–cost–latency scatter plot;
- a radar comparison for up to three selected candidates;
- model-level answer and metadata panels;
- expandable agent evidence and uncertainty cards;
- an embedded MLflow run summary;
- an OpenTelemetry trace waterfall and span table.

### Score direction

To avoid confusing risk and quality metrics:

- raw **bias** and **toxicity** agent scores are risk scores, where lower is better;
- the UI converts them to **bias safety** and **toxicity safety**, where higher is better;
- **faithfulness**, **reasoning**, and **overall** scores are quality scores, where higher is better.

---

## Evidence exports

Every benchmark can produce one complete JSON evidence bundle containing:

- benchmark ID and timestamp;
- question and optional context;
- candidate, evaluator, and judge configuration;
- randomized blind execution order;
- generated answers;
- all agent scores, labels, evidence, explanations, and uncertainty;
- token usage, reasoning-token usage, latency, and estimated API cost;
- MLflow parent and child run IDs;
- OpenTelemetry trace ID and finished spans;
- candidate failures, when applicable.

Additional downloads include:

- MLflow run manifest JSON;
- OpenTelemetry trace JSON;
- comparison table CSV.

The API key is never included in reports, traces, MLflow parameters, or downloadable files.

---

## Technology stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit + scoped CSS | Interactive benchmark configuration, prominent result tabs, visualization, and downloads |
| LLM API | OpenAI Python SDK + Responses API | Candidate generation and evaluator/judge calls |
| Structured output | OpenAI JSON Schema output + Pydantic | Reliable rubric output parsing and report validation |
| Data processing | Pandas | Leaderboard and export tables |
| Visualization | Plotly | Heatmap, radar, quality–cost scatter, MLflow chart, trace waterfall |
| Experiment tracking | MLflow | Parent benchmark run, candidate child runs, metrics, parameters, and artifacts |
| Observability | OpenTelemetry SDK | Root benchmark trace and spans for generation, evaluators, judge, and MLflow logging |
| Configuration | JSON + python-dotenv + Streamlit secrets | Curated model catalog and server-side credential configuration |
| Testing | Pytest | Offline catalog and preset validation |

### Recommended Python version

Python 3.10, 3.11, or 3.12.

---

## Project structure

```text
agentic-llm-eval/
├── app.py
├── run_batch.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── agents/
│   ├── base.py
│   ├── bias_agent.py
│   ├── toxicity_agent.py
│   ├── faithfulness_agent.py
│   ├── reasoning_agent.py
│   └── judge_agent.py
├── core/
│   ├── answer_generator.py
│   ├── config.py
│   ├── llm_client.py
│   ├── model_catalog.py
│   ├── orchestrator.py
│   └── schemas.py
├── data/
│   ├── models.json
│   └── sample_inputs.json
├── tracking/
│   ├── mlflow_logger.py
│   └── telemetry.py
├── ui/
│   ├── styles.py
│   └── visualizations.py
└── tests/
    └── test_catalog.py
```

---

# Run locally in VS Code

## 1. Open the project

Extract the ZIP and open the `agentic-llm-eval` folder:

```text
VS Code -> File -> Open Folder
```

Open a terminal in VS Code:

```text
Terminal -> New Terminal
```

Make sure the terminal is inside the folder containing `app.py`.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
python -m venv venv
venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure the OpenAI key

Create `.env` from the example.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS or Linux

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```text
OPENAI_API_KEY=your_real_key_here
```

The `.env` file must be next to `app.py`. It is already excluded by `.gitignore`.

The application loads `.env` directly using an absolute project path, so VS Code's `python.terminal.useEnvFile` setting is not required for the Streamlit app itself.

The application does not expose an API-key input in the user interface. Key resolution is automatic:

1. Streamlit secrets when deployed;
2. `.env` or an operating-system environment variable during local development.

## 5. Start the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

The sidebar is headed **Evaluation Configuration** and shows whether the server-side API key was detected. The key itself is never displayed or requested from the user.

## 6. Run tests

These tests do not call the OpenAI API:

```bash
pytest -q
```

## 7. Optional batch benchmark

```bash
python run_batch.py
```

The batch runner reads examples from `data/sample_inputs.json`, uses the three default candidate configurations, logs to MLflow, and saves JSON evidence reports.

---

# MLflow experiment tracking

The workbench creates:

- one **parent run** per benchmark session;
- one nested **candidate run** per successful candidate configuration.

Each candidate run logs:

- candidate configuration and reasoning effort;
- evaluator and judge models;
- prompt version and preset;
- bias risk and safety;
- toxicity risk and safety;
- faithfulness and reasoning quality;
- overall score and verdict;
- latency, token usage, reasoning tokens, and estimated cost;
- generated-answer and evaluation-report artifacts.

Start the MLflow UI in a second terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

The Streamlit app also shows an embedded MLflow summary, so a viewer can understand the experiment structure without opening the external interface.

---

# OpenTelemetry tracing

The application records a root `benchmark_request` trace with nested spans such as:

```text
benchmark_request
├── candidate_benchmark
│   ├── generate_answer
│   ├── bias_agent
│   ├── toxicity_agent
│   ├── faithfulness_agent
│   ├── reasoning_agent
│   └── judge_agent
└── mlflow_logging
```

Finished spans are captured in memory and shown as a waterfall chart. They are also included in the JSON evidence bundle.

To additionally print raw spans in the terminal, set:

```text
OTEL_CONSOLE_EXPORTER=true
```

A future production deployment could replace or complement the in-memory exporter with an OTLP Collector, Jaeger, Grafana Tempo, or another compatible backend.

---

# How the cost estimate works

The OpenAI Responses API returns input and output token usage. The application multiplies those counts by the model prices configured in `data/models.json`:

```text
estimated cost =
(input tokens / 1,000,000 × input price)
+
(output tokens / 1,000,000 × output price)
```

Reasoning tokens are included in the API's output token count and are also recorded separately when the response exposes them.

The estimate does not include possible regional surcharges, cached-input discounts, batch discounts, or future pricing changes.

---




---

## Official references

- OpenAI model documentation: https://developers.openai.com/api/docs/models
- OpenAI Responses API: https://developers.openai.com/api/docs/guides/text
- OpenAI reasoning guide: https://developers.openai.com/api/docs/guides/reasoning
- MLflow documentation: https://mlflow.org/docs/latest/
- OpenTelemetry Python documentation: https://opentelemetry.io/docs/languages/python/
- Streamlit documentation: https://docs.streamlit.io/
