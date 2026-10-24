# AMB-Sentinel (repository: AMB-APP)

An autonomous, local-first AI agent service. It reads a task **instruction** from `config.yaml`, lets a language model plan and call tools (web search, scraping, feeds, downloads, text extraction, transcription, database access), stores what it finds in PostgreSQL, and analyses the stored items with the model on a schedule. It can use a local GGUF model directly, the **Local LLM Cloud** server (see the [Local-LLM repository](https://github.com/Deveeelloopase/Local-LLM)), Hugging Face, GPT4All or an OpenAI-compatible endpoint.

The code and logs call the system **AMB-Sentinel**; `AMB-APP` is the repository name.

## What runs

```
config.yaml ──▶ ai_agent.py  (AIModelService)
                 ├─ LLM adapters: local_llm_cloud · llama_cpp · hf · gpt4all · openai   (auto-fallback chain)
                 ├─ AgentExecutor ── instruction → goals → tool calls ──▶ tools/*  (154 registered callables)
                 ├─ DataAnalyzer ── reads raw_data, writes analysis_results (periodic)
                 ├─ PostgreSQL (+pgvector) and Redis
                 └─ APScheduler jobs: health_check, model_warming (5 min), memory_cleanup (24 h), metrics_collection (1 min)
```

Everything is in-process; there is no HTTP API. `ai_service.enable_monitoring` (default true) and `ai_service.metrics_port` (default 9090) configure monitoring.

## Repository contents

| Path | Purpose |
|---|---|
| `ai_agent.py` | The application (~13,000 lines): configuration dataclasses, LLM adapters, prompt/parsing helpers, `AgentExecutor`, `DataAnalyzer`, `ModelFactory`, `AIModelService`, `main()` |
| `llm_cloud_adapter.py` | HTTP client for the Local LLM Cloud server (`LLMCloudClient`, `probe_server()`, `get_client()`), circuit breaker, typed errors |
| `model_manager.py` | Hardware-aware context/token budgeting (`AdaptiveModelManager`, `AdaptiveTokenBudget`, `GenerationTelemetry`) and `BackendOrchestrator`, which picks the active LLM backend |
| [`tools/`](src/amb_sentinel/tools/README.md) | The tool modules the agent can call |
| `test_local_model.py` | Interactive script to try a GGUF model with `llama-cpp-python` and print speed metrics |
| `setup.sh` | Idempotent installer (system packages, virtualenv, Python packages, PostgreSQL + pgvector, Playwright browsers, optional systemd service `amb-sentinel`) |
| `secure_local.sh` | One-time permission hardening for `config.yaml`, logs, model directory |
| `requirements.txt` | Python dependencies |
| `CHANGELOG.md` | History of the v5.0.0 upgrade |
| `.gitignore` | Ignores `config.yaml`, `logs/`, `*.log`, etc. |
| `.idea/` | JetBrains IDE metadata committed by accident — safe to delete and git-ignore |

Files that exist only on a deployed machine (git-ignored): `config.yaml`, `logs/`.

## Install

```bash
git clone https://github.com/michealbamidele147-cmd/AMB-APP.git && cd AMB-APP
./setup.sh --help                    # options: --venv PATH --config PATH --install-service --force --dry-run --no-venv
./setup.sh                           # installs system + Python dependencies, PostgreSQL/pgvector, Playwright
# create config.yaml (see "Configuration"), then:
./secure_local.sh                    # optional: lock down config, logs, models
```

## Run

```bash
python ai_agent.py --config config.yaml                    # service: scheduler + agent loop + analyzer
python ai_agent.py --config config.yaml --run-agent-once   # run the agent once with the configured instruction, then exit
python ai_agent.py --config config.yaml --health-check     # check components and exit
python ai_agent.py --config config.yaml --no-agent         # services only, no automatic agent runs
python ai_agent.py --log-level DEBUG
```

With `--install-service`, `setup.sh` registers the systemd unit `amb-sentinel`.

## Configuration (`config.yaml`)

The file is not in the repository. These are the sections the code reads:

| Section | Keys |
|---|---|
| `ai_service` | `models` (list), `primary_model`, `agent_instruction`, `auto_run_agent`, `max_agent_iterations`, `max_total_tokens`, `agent_execution_interval`, `data_analysis_interval`, `enable_concurrent_execution`, `health_check_interval`, `max_concurrent_requests`, `request_timeout`, `download_dir`, `enable_monitoring`, `metrics_port`, `embeddings:` (`model`, `cache_dir`, `cache_size`, `enable_download`, `timeout`, `device`, `similarity_threshold`), `validator:` (`forbidden_patterns`) |
| `ai_service.models[]` | `provider`, `model_name`, `enabled`, `api_key`, `base_url`, `temperature`, `max_tokens`, `timeout`, `device`, `max_retries`, `quantization`, `context_length`, `batch_size`, `trust_remote_code`, `low_cpu_mem_usage`, `n_gpu_layers`, `n_threads`, `chat_template` |
| `database.postgresql` | `host`, `port`, `database`, `user`, `password`, `min_connections`, `max_connections`, `timeout`, `sslmode`, `pgvector:` (`enabled`, `dimension`, `index_type`, `lists`, `probes`, `operator`) |
| `database.redis` | `host`, `port`, `db`, `password`, `max_connections`, `socket_timeout`, `socket_connect_timeout`, `retry_on_timeout` |
| `data_analyzer` | `domain`, `interval`, `batch_size`, `max_concurrent`, `circuit_breaker_threshold`, `circuit_breaker_cooldown`, `prompt_template`, `output_schema`, `min_confidence`, and table/column names (`raw_data_table`, `raw_data_id_col`, `raw_data_source_col`, `raw_data_content_col`, `raw_data_analyzed_col`, `raw_data_order_col`, `analysis_results_table`) |

`provider` values: `local_llm_cloud` (aliases `llm_cloud`, `cloud`), `llama_cpp` (`llama`, `llamacpp`, `llama-cpp`, `llama.cpp`), `hf` (`huggingface`, `transformers`), `gpt4all`, `openai`, or `auto`. In `auto` / `local_llm_cloud` mode the factory tries: Local LLM Cloud → llama-cpp → Hugging Face → OpenAI-compatible → a dummy adapter.

Use environment variables for secrets rather than literals: `LLM_CLOUD_API_KEY`, `LLM_CLOUD_HOST`, `LLM_CLOUD_PORT`, `LLM_CLOUD_MODEL_ID`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `REDIS_PASSWORD`, `POSTGRES_PASSWORD` / `PGPASSWORD` / `DB_PASSWORD` are read by the code.

## Using Local LLM Cloud

```yaml
ai_service:
  primary_model: local_llm_cloud
  models:
    - provider: local_llm_cloud
      model_name: qwen3-8b          # an id from GET /v1/models on the server
      base_url: http://localhost:8080
      api_key: ${LLM_CLOUD_API_KEY}
```

The adapter uses the non-streaming `/v1/chat/completions` endpoint of the server. Things that matter for it to work reliably:

- **Send a deadline.** Pass `timeout_s` in every request (the server treats a missing value as its `inference.default_timeout_s`). the runtime modules described below do this.
- **Own the history.** Send `history_mode: "replace"` and a unique `X-Session-ID`, so the server never appends stored turns to an agent prompt.
- **Use the server's limits.** Read `GET /v1/runtime/limits` (`context_length`, `effective_tps`) rather than assuming values.
- **No native tool calling.** The server ignores `tools`; the agent asks the model for a JSON action in plain text.
- **Reasoning models.** Send `no_think: true` instead of appending `<think></think>` to the prompt.

### LLM runtime modules

Three modules in the repository root implement the points above. They are used by the agent loop and the Local-LLM adapter:

| Module | Contents |
|---|---|
| `amb_llm_core.py` | `ChatClient(base_url, api_key, model)`: `ensure_model()` (resolves the configured name against `/v1/models`), `runtime_limits()`, `chat(messages, …)`, `chat_stream(messages, …)` (first-token and idle watchdogs). Typed errors: `LLMError` → `LLMAuthError`, `LLMContextOverflow`, `LLMTransportError`, `LLMModelNotReady`, `LLMServerTimeout` (`LLMStall`), `LLMBadReply`. `validate_reply()` rejects error text that a server returns inside a 200 reply (for example `[timed out after 0s]`); `estimate_deadline_s()` sizes the deadline from prompt tokens, output tokens and measured speed |
| `amb_actions.py` | `ToolSpec.from_callable()` builds a compact tool card from a function signature; `parse_action(text, specs)` turns a model reply into a validated `Action` (`tool` / `final` / `none`): strips `<think>`, finds JSON at any nesting depth, matches the tool name only on the `tool` field, coerces arguments by declared type, and produces a repair hint for missing arguments |
| `amb_runtime.py` | `CapabilityProfile.build()` (tier `nano`/`small`/`medium`/`large` from context size, model size and speed), `pack()` (fit prompt sections into a token budget), `compile_instruction()` / `segment_step()` (split the configured instruction into steps), `Blackboard` (tool results addressed by handles like `$r1.links.0`), and `AgentRun` (step-by-step loop with repair turns, duplicate-call memo and tier downgrade) |

Tests for these modules ship with the playbook archive (`amb_ref.tar.gz`: `tests/test_llm_core.py`, `test_actions.py`, `test_runtime.py`); run them with `pytest`.

## Database

Created on first start (`CREATE TABLE IF NOT EXISTS`): `agent_executions`, `raw_data`, `analysis_results`, `tool_executions`, `embeddings` (pgvector). Tool modules can also create their own tables. Redis is optional (`database.redis`).

## Logs

Each component writes its own file under `logs/`: `ai_agent.log`, `agent_executor.log`, `data_analyzer.log`, `apscheduler.log`, `registrar.log`, `system_metric.log`, `modelmanager.log`, plus one per tool module (`scrape.log`, `http.log`, `findlink.log`, `download.log`, `extract.log`, `transcribe.log`, `video.log`, `pipeline.log`, `dbtools.log`, `utils.log`, `initials.log`, `main.log`). A tool log that stays empty means that tool was never called.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Agent log shows `Response contains no JSON … '[timed out after 0s]'` repeatedly | The model server returned a deadline error inside a 200 reply. Make sure the request carries `timeout_s` and the server treats `default_timeout_s: 0` as "no deadline" |
| `local-llm-cloud not reachable` then `Creating emergency DummyAdapter` at start-up | The server was not up yet. Start it first, or let the service wait for it; do not run production on the dummy adapter |
| Agent stops after one or two iterations | `ai_service.max_total_tokens` is a run budget; if it is smaller than a couple of prompts the run ends early |
| `scrape.log` and other tool logs empty | No tool call has happened — look at `agent_executor.log` for LLM errors |

## Security notes

- Never commit `config.yaml` (it holds database and API credentials); it is git-ignored for that reason. `secure_local.sh` restricts its permissions.
- `dbtools.py` masks passwords in logged URIs and has a guard that blocks potentially dangerous SQL statements; the log line `Blocked potentially dangerous SQL: ALTER operation` appears at start-up (the statement that triggers it has not been identified).
- The agent fetches arbitrary web content; run it under a dedicated user and respect the target sites' terms and robots rules (`tools/scrape.py` has a robots policy setting).

## Known issues in the repository state

- `ai_agent.py` contains duplicated method definitions inside `AgentExecutor` and unused prompt-engine code; later definitions silently override earlier ones.
- Several paths in code and config examples are specific to the author's machine.
