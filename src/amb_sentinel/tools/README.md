# tools/

The capabilities the agent can use. Every module is a set of plain Python functions (sync or async) plus supporting classes. At start-up `registrar.py` registers them as tools in the agent's `ToolRegistry`, using each function's name, signature and docstring.

| Module | Purpose | Main tool functions |
|---|---|---|
| `__init__.py` | Package facade: re-exports tool modules, availability probes, `initialize_tools()`, `get_tools_status()`, `get_full_system_health()` | `rss_fetch`, `google_news_search`, `scrape_multiple`, backend listers |
| `registrar.py` | `ToolsRegistrar`: registers all tool functions with wrappers (argument checking, logging); `create_enhanced_tool_wrapper()` | – |
| `http.py` | HTTP client with retries, rate limiting, circuit breaker, in-flight de-duplication and size/security limits | `http_fetch`, `http_fetch_async`, `http_fetch_batch`, `get_http_metrics` |
| `utils.py` | Foundation helpers: URL validation, safe file names, SSL contexts, WAF-friendly headers, user-agent rotation, session pool, response cache, per-domain adaptive delay, file hashing, downloads, cleanup | `validate_url`, `sanitize_filename`, `make_session`, `download_file`, `download_file_async`, `cleanup_downloads` |
| `find_links.py` | Link and URL discovery with engine scoring | `search_links`, `search_with_dorks`, `find_file_links`, `find_hidden_resources`, `extract_links_from_page`, `expand_links`, `rank_links` |
| `scrape.py` | Web scraping (requests/BeautifulSoup, Playwright for JS pages), RSS/Atom feeds, robots policy, content de-duplication, cache | `scrape_page`, `parse_feed`, `scrape_batch`, `scrape_async`, `scrape_batch_async`, `browser_scrape` |
| `download.py` | File download manager with integrity checks, priorities and per-domain rate limiting | `download_file`, `download_multiple`, `verify_file_integrity`, `calculate_multiple_hashes` |
| `extract.py` | Text extraction from PDF, DOCX, spreadsheets/CSV, text and images (OCR); chunking | `extract_text_from_file`, `extract_text_from_pdf`, `extract_text_with_ocr`, `chunk_text`, `batch_extract_files`, `process_file_for_vector_store` |
| `transcribe.py` | Audio/video → text through a priority chain of backends starting with faster-whisper | `transcribe_audio_or_video`, `transcribe_multiple`, `clear_model_cache` |
| `video.py` | Online video metadata and download via yt-dlp, with generic HTTP and ffmpeg fallbacks | `video_info`, `download_video`, `download_multiple_videos`, `extract_video_thumbnail` |
| `pipeline.py` | Ingest → transcribe/extract → chunk → embed → index pipeline with checkpoints | `transcribe_url_to_vectorstore`, `transcribe_and_index`, `register_pipeline_tools` |
| `dbtools.py` | Database access for the agent: connection management, query cache and metrics, bulk insert, streaming queries, SQL safety checks, masked URIs in logs | registered by `register_db_tools()` (e.g. `db_bulk_insert`, `db_get_metrics`, `db_stream_query`, `db_invalidate_cache`) |

## Rules for writing a tool

1. Give each parameter a **type hint** and the function a one-line **docstring**: they become the tool card the model sees (`name(arg*, opt:int=5) — description`).
2. Avoid `**kwargs` in tool signatures; the model cannot know parameters that are not declared.
3. Return JSON-serialisable data. Put errors in the return value (`{"error": "..."}`) or raise; do not return partial success silently.
4. Keep results compact. Large payloads should be stored (database, file) and returned as a short summary plus an identifier.
5. Log to the module's own logger so `logs/<module>.log` shows what ran.
6. Register the function in `registrar.py` (and export it from `__init__.py` if it is public).

## Notes

- The adaptive context manager (`model_manager.py`) lives next to `ai_agent.py` at the repository root, not here; tool modules import it with `from model_manager import …`.
- Heavy dependencies are optional and probed at import time (`playwright`, `yt-dlp`, `faster-whisper`, `pytesseract`, `pdfminer`, …); the matching tool reports itself unavailable instead of crashing the service.
- 154 callables are registered in total, including helper constructors; helpers whose names start with `create_`, `check_` or `get_default_` are not useful to a model and are best hidden from the tool list shown to it.
