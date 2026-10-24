"""
amb_actions.py — production tool schema + model-output -> validated action.

A pure, deterministic module. No I/O, no LLM, no global state. Everything here
is a pure function of its inputs, so it is trivially testable and cannot hang.

Responsibilities
────────────────
  * Describe tools to the LLM (compact cards, examples, JSON Schema)
  * Parse the LLM's reply into a validated Action, tolerant on input and
    strict on output
  * Never execute a tool because its name appeared in prose
  * Coerce argument values by the tool's declared types, not by guesses
  * Resolve "$r1.links.0" handles against a caller-supplied resolver, so the
    model never has to copy long values
  * Produce a one-line repair hint when parsing fails
"""
from __future__ import annotations

import difflib
import inspect
import json
import re
import typing
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union,
)

T = TypeVar("T")


# ═════════════════════════════════════════════════════════════════════════════
# Param / ToolSpec
# ═════════════════════════════════════════════════════════════════════════════
_PY_TO_JSON: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}

_JSON_TO_PY: Dict[str, Tuple[type, ...]] = {
    "string":  (str,),
    "integer": (int,),
    "number":  (int, float),
    "boolean": (bool,),
    "array":   (list, tuple, set),
    "object":  (dict,),
    "null":    (type(None),),
}

_DEFAULT_EXAMPLES: Dict[str, Any] = {
    "url": "https://example.org/page",
    "uri": "https://example.org/page",
    "query": "search terms",
    "q": "search terms",
    "path": "/path/to/file",
    "file_path": "/path/to/file",
    "dest_dir": "/tmp/downloads",
    "filename": "file.txt",
    "text": "some text",
    "sql": "SELECT 1",
    "language": "eng",
    "model_name": "small",
    "pattern": "regex",
    "site": "example.org",
    "table": "raw_data",
    "columns": ["col_a", "col_b"],
    "rows": [["v1", "v2"]],
    "urls": ["https://example.org/a", "https://example.org/b"],
    "paths": ["/path/to/a", "/path/to/b"],
    "feeds": ["https://example.org/feed.xml"],
    "tags": ["alpha", "beta"],
}


@dataclass
class Param:
    """
    One parameter of a tool.

    `type` is a JSON Schema type name (string / integer / number / boolean /
    array / object / null). `default` is the declared Python default; when
    `has_default` is True the runtime may inject it for optional params.

    `doc` is the one-line human description, extracted from the function's
    docstring when available.
    """
    name: str
    type: str = "string"
    required: bool = True
    default: Any = None
    doc: str = ""
    has_default: bool = False
    enum: Optional[List[Any]] = None     # optional enumerated values


@dataclass
class ToolSpec:
    """
    Compact description of one callable tool.

    `params` is keyed by parameter name (order preserved). `aliases` are
    alternative names the model may use in the "tool" field. `accepts_extra`
    means the underlying function has **kwargs, so unknown args are passed
    through rather than dropped.
    """
    name: str
    description: str = ""
    params: Dict[str, Param] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    accepts_extra: bool = False

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_callable(
        cls,
        name: str,
        fn: Callable,
        description: str = "",
        aliases: Tuple[str, ...] = (),
    ) -> "ToolSpec":
        """
        Build a ToolSpec from a Python callable.

        * Type hints drive the parameter types (Optional[X] -> X).
        * Defaults drive required/optional.
        * **kwargs sets accepts_extra.
        * Docstring Args block is mined for param descriptions and — for
          **kwargs tools — for otherwise-invisible parameters.
        """
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return cls(name, description or (inspect.getdoc(fn) or ""), {}, aliases, False)

        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}

        doc = inspect.getdoc(fn) or ""
        doc_map = _parse_docstring_args(doc)

        params: Dict[str, Param] = {}
        extra = False
        for pn, p in sig.parameters.items():
            if pn in ("self", "cls"):
                continue
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                extra = True
                continue
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                continue

            t = hints.get(pn, str)
            # Unwrap Optional[X] -> X
            origin = typing.get_origin(t)
            if origin in (typing.Union, Union):
                inner = [a for a in typing.get_args(t) if a is not type(None)]
                if inner:
                    t = inner[0]
                    origin = typing.get_origin(t)

            base = origin or t
            jtype = _PY_TO_JSON.get(base, "string")
            has_default = p.default is not inspect.Parameter.empty
            default = None if not has_default else _safe_default(p.default)

            params[pn] = Param(
                name=pn,
                type=jtype,
                required=not has_default,
                default=default,
                doc=(doc_map.get(pn, ("", ""))[1] or "")[:80],
                has_default=has_default,
            )

        # For **kwargs tools, mine the docstring for extra parameter names.
        if extra and doc_map:
            for pn, (jtype, pdoc) in doc_map.items():
                if pn in params:
                    continue
                if pn.lower() in ("args", "kwargs", "returns", "raises", "self", "cls"):
                    continue
                params[pn] = Param(
                    name=pn,
                    type=jtype or "string",
                    required=False,
                    default=None,
                    doc=(pdoc or "")[:80],
                    has_default=False,
                )

        first = (description or doc).strip().split("\n", 1)[0].strip()
        return cls(name, first[:180], params, aliases, extra)

    # ── rendering ───────────────────────────────────────────────────────────
    def card(self, max_chars: int = 150) -> str:
        """
        One-line tool card:
            name(url*, max_entries:int=100) — what it does

        `*` marks a required parameter. Type names are abbreviated. Defaults
        are shown for scalars only (list/dict defaults are omitted to keep the
        line short).
        """
        parts: List[str] = []
        for p in self.params.values():
            s = p.name + ("*" if p.required else "")
            if p.type != "string":
                s += ":" + {
                    "integer": "int", "number": "float",
                    "boolean": "bool", "array": "list", "object": "dict",
                }.get(p.type, p.type)
            if (
                not p.required
                and p.has_default
                and isinstance(p.default, (int, float, bool, str))
                and p.default not in ("", None)
            ):
                s += f"={json.dumps(p.default)}"
            parts.append(s)
        head = f"{self.name}({', '.join(parts)})"
        if not self.description:
            return head[:max_chars]
        room = max_chars - len(head) - 3
        if room < 8:
            return head[:max_chars]
        desc = self.description[:room]
        return f"{head} — {desc}"

    def example(self) -> str:
        """
        A plausible call for this tool, derived from parameter names and types.

        Never emits a hardcoded domain-specific URL. Uses `_DEFAULT_EXAMPLES`
        when the parameter name is known, otherwise a type-appropriate value.
        """
        args: Dict[str, Any] = {}
        for p in self.params.values():
            if not p.required:
                continue
            if p.name in _DEFAULT_EXAMPLES:
                args[p.name] = _DEFAULT_EXAMPLES[p.name]
            else:
                args[p.name] = _value_for_type(p.type)
        return json.dumps({"tool": self.name, "args": args}, ensure_ascii=False)

    def json_schema(self) -> Dict[str, Any]:
        """OpenAPI 3.1 / OpenAI tools-compatible schema for this tool."""
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in self.params.values():
            props[p.name] = {"type": p.type}
            if p.doc:
                props[p.name]["description"] = p.doc
            if p.enum:
                props[p.name]["enum"] = list(p.enum)
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": bool(self.accepts_extra),
                },
            },
        }


def _safe_default(v: Any) -> Any:
    """Return a JSON-serialisable default, or None for complex objects."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return list(v)[:8]
    if isinstance(v, dict):
        return {k: _safe_default(x) for k, x in list(v.items())[:8]}
    return None


def _value_for_type(jtype: str) -> Any:
    return {
        "string": "value",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": ["value"],
        "object": {"key": "value"},
        "null": None,
    }.get(jtype, "value")


# ── docstring Args-block parser ───────────────────────────────────────────
_DOC_TYPE_ALIAS = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "double": "number",
    "bool": "boolean", "boolean": "boolean", "flag": "boolean",
    "list": "array", "array": "array", "sequence": "array",
    "dict": "object", "object": "object", "mapping": "object",
    "none": "null", "null": "null",
}


def _parse_docstring_args(doc: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse the Args/Parameters block of a docstring.

    Returns {param_name: (json_type, description)}. Handles:
        Args:
            name (str): description
            count (int, optional): description
        Parameters
        ----------
        name : str
            description
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not doc:
        return out

    # Google-style: "Args:" or "Arguments:" block
    g = re.search(
        r"^\s*(?:Args|Arguments|Parameters)\s*:\s*\n((?:[ \t]+\S.*\n?)+)",
        doc,
        re.M,
    )
    if g:
        for line in g.group(1).splitlines():
            m = re.match(
                r"^\s*(\*{0,2}\w+)\s*(?:\(\s*([^)]+)\s*\))?\s*:\s*(.+)$",
                line,
            )
            if not m:
                continue
            pname = m.group(1).lstrip("*")
            typ = (m.group(2) or "").split(",")[0].strip().lower()
            jtype = _DOC_TYPE_ALIAS.get(typ, "string")
            out[pname] = (jtype, m.group(3).strip())

    # NumPy-style:
    #   Parameters
    #   ----------
    #   name : str
    #       description
    if not out:
        n = re.search(
            r"^\s*Parameters\s*\n\s*-{3,}\s*\n((?:(?:\s*\S.*\n)(?:\s{4,}\S.*\n)*)+)",
            doc,
            re.M,
        )
        if n:
            for m in re.finditer(
                r"^\s*(\w+)\s*:\s*([\w\[\], ]+)\s*\n\s+([^\n]+)",
                n.group(1),
                re.M,
            ):
                pname = m.group(1)
                typ = m.group(2).split(",")[0].strip().lower()
                jtype = _DOC_TYPE_ALIAS.get(typ, "string")
                out[pname] = (jtype, m.group(3).strip())

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Explicit overrides for tools whose real signature is hidden behind **kwargs
# ═════════════════════════════════════════════════════════════════════════════
#
# The registrar wraps many tools so their real parameters are not visible via
# inspect.signature(). Without an override the LLM sees an empty parameter
# list and cannot call the tool correctly.
#
# Only include tools here when from_callable would produce an incomplete or
# misleading spec. Naming and ordering follow the registrar so grep works.
TOOL_SPEC_OVERRIDES: Dict[str, ToolSpec] = {
    # ── HTTP ─────────────────────────────────────────────────────────────
    "http_fetch": ToolSpec(
        name="http_fetch",
        description="Fetch a URL over HTTP(S) and return status, headers, and body.",
        params={
            "url":     Param("url", "string", True),
            "method":  Param("method", "string", False, "GET", has_default=True,
                             enum=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]),
            "headers": Param("headers", "object", False, None),
            "timeout": Param("timeout", "integer", False, 30, has_default=True),
        },
    ),
    "http_fetch_batch": ToolSpec(
        name="http_fetch_batch",
        description="Fetch multiple URLs concurrently and return results in order.",
        params={
            "urls":           Param("urls", "array", True),
            "method":         Param("method", "string", False, "GET", has_default=True),
            "max_concurrent": Param("max_concurrent", "integer", False, 10, has_default=True),
            "timeout":        Param("timeout", "integer", False, 30, has_default=True),
        },
    ),
    # ── Download ─────────────────────────────────────────────────────────
    "download_file": ToolSpec(
        name="download_file",
        description="Download a single file from a URL to a local directory.",
        params={
            "url":         Param("url", "string", True),
            "dest_dir":    Param("dest_dir", "string", False, None),
            "filename":    Param("filename", "string", False, None),
            "timeout":     Param("timeout", "integer", False, 120, has_default=True),
            "max_retries": Param("max_retries", "integer", False, 3, has_default=True),
            "resume":      Param("resume", "boolean", False, True, has_default=True),
            "verify_ssl":  Param("verify_ssl", "boolean", False, True, has_default=True),
        },
    ),
    "download_multiple": ToolSpec(
        name="download_multiple",
        description="Download multiple files concurrently.",
        params={
            "urls":           Param("urls", "array", True),
            "dest_dir":       Param("dest_dir", "string", False, None),
            "max_concurrent": Param("max_concurrent", "integer", False, 8, has_default=True),
            "timeout":        Param("timeout", "integer", False, 120, has_default=True),
        },
    ),
    # ── Scraping ─────────────────────────────────────────────────────────
    "browser_scrape": ToolSpec(
        name="browser_scrape",
        description="Scrape a page with a headless browser (for JS-heavy sites).",
        params={
            "url":       Param("url", "string", True),
            "engine":    Param("engine", "string", False, "playwright", has_default=True,
                               enum=["playwright", "http"]),
            "wait_time": Param("wait_time", "integer", False, 5, has_default=True),
        },
    ),
    "parse_feed": ToolSpec(
        name="parse_feed",
        description="Parse a single RSS/Atom feed and return its entries.",
        params={
            "url":           Param("url", "string", True),
            "max_entries":   Param("max_entries", "integer", False, 100, has_default=True),
            "parse_entries": Param("parse_entries", "boolean", False, False, has_default=True),
        },
    ),
    "parse_feeds_batch": ToolSpec(
        name="parse_feeds_batch",
        description="Parse multiple RSS/Atom feeds in parallel.",
        params={
            "feeds":         Param("feeds", "array", True),
            "max_entries":   Param("max_entries", "integer", False, 100, has_default=True),
            "parse_entries": Param("parse_entries", "boolean", False, True, has_default=True),
        },
    ),
    # ── Extraction ───────────────────────────────────────────────────────
    "extract_text_from_file": ToolSpec(
        name="extract_text_from_file",
        description="Extract plain text from any supported file (auto-detects format).",
        params={
            "path":         Param("path", "string", True),
            "method":       Param("method", "string", False, "auto", has_default=True),
            "max_pages":    Param("max_pages", "integer", False, None),
            "ocr_enabled":  Param("ocr_enabled", "boolean", False, False, has_default=True),
        },
    ),
    "extract_text_from_pdf": ToolSpec(
        name="extract_text_from_pdf",
        description="Extract text from a PDF file.",
        params={
            "path":              Param("path", "string", True),
            "max_pages":         Param("max_pages", "integer", False, None),
            "method":            Param("method", "string", False, "auto", has_default=True),
            "preserve_layout":   Param("preserve_layout", "boolean", False, False, has_default=True),
            "extract_metadata":  Param("extract_metadata", "boolean", False, True, has_default=True),
        },
    ),
    "extract_text_from_docx": ToolSpec(
        name="extract_text_from_docx",
        description="Extract text from a DOCX file.",
        params={
            "path":                    Param("path", "string", True),
            "include_comments":        Param("include_comments", "boolean", False, False, has_default=True),
            "include_headers_footers": Param("include_headers_footers", "boolean", False, False, has_default=True),
        },
    ),
    "extract_text_from_spreadsheet": ToolSpec(
        name="extract_text_from_spreadsheet",
        description="Extract text from a spreadsheet (XLSX/XLS/ODS/CSV).",
        params={
            "path":             Param("path", "string", True),
            "sheet":            Param("sheet", "string", False, None),
            "nrows":            Param("nrows", "integer", False, None),
            "include_headers":  Param("include_headers", "boolean", False, True, has_default=True),
        },
    ),
    "extract_text_from_textfile": ToolSpec(
        name="extract_text_from_textfile",
        description="Extract text from a plain text file (TXT/MD/LOG/...).",
        params={
            "path":             Param("path", "string", True),
            "encoding":         Param("encoding", "string", False, "utf-8", has_default=True),
            "max_chars":        Param("max_chars", "integer", False, None),
            "detect_encoding":  Param("detect_encoding", "boolean", False, True, has_default=True),
        },
    ),
    "extract_with_ocr": ToolSpec(
        name="extract_with_ocr",
        description="Extract text from an image using OCR (Tesseract).",
        params={
            "path":     Param("path", "string", True),
            "language": Param("language", "string", False, "eng", has_default=True),
            "psm":      Param("psm", "integer", False, 3, has_default=True),
            "enhance":  Param("enhance", "boolean", False, True, has_default=True),
        },
    ),
    "extract_and_chunk_file": ToolSpec(
        name="extract_and_chunk_file",
        description="Extract text from a file, then split it into overlapping chunks.",
        params={
            "path":                Param("path", "string", True),
            "chunk_size":          Param("chunk_size", "integer", False, 1000, has_default=True),
            "chunk_overlap":       Param("chunk_overlap", "integer", False, 200, has_default=True),
            "chunking_strategy":   Param("chunking_strategy", "string", False, "sentence", has_default=True),
        },
    ),
    "batch_extract_files": ToolSpec(
        name="batch_extract_files",
        description="Extract text from multiple files in parallel.",
        params={
            "file_paths":     Param("file_paths", "array", True),
            "max_concurrent": Param("max_concurrent", "integer", False, 8, has_default=True),
        },
    ),
    "chunk_text": ToolSpec(
        name="chunk_text",
        description="Split plain text into overlapping chunks.",
        params={
            "text":            Param("text", "string", True),
            "chunk_size":      Param("chunk_size", "integer", False, 1000, has_default=True),
            "chunk_overlap":   Param("chunk_overlap", "integer", False, 200, has_default=True),
            "strategy":        Param("strategy", "string", False, "sentence", has_default=True),
        },
    ),
    # ── Video ────────────────────────────────────────────────────────────
    "download_video": ToolSpec(
        name="download_video",
        description="Download a video from a supported platform (yt-dlp).",
        params={
            "url":        Param("url", "string", True),
            "dest_dir":   Param("dest_dir", "string", False, "downloads/video", has_default=True),
            "quality":    Param("quality", "string", False, "best", has_default=True),
            "audio_only": Param("audio_only", "boolean", False, False, has_default=True),
            "format":     Param("format", "string", False, "mp4", has_default=True),
            "filename":   Param("filename", "string", False, None),
            "timeout":    Param("timeout", "integer", False, 3600, has_default=True),
        },
    ),
    "download_multiple_videos": ToolSpec(
        name="download_multiple_videos",
        description="Download multiple videos in parallel.",
        params={
            "urls":           Param("urls", "array", True),
            "dest_dir":       Param("dest_dir", "string", False, "downloads/video", has_default=True),
            "max_concurrent": Param("max_concurrent", "integer", False, 4, has_default=True),
            "quality":        Param("quality", "string", False, "best", has_default=True),
        },
    ),
    # ── Transcription ────────────────────────────────────────────────────
    "transcribe_audio_or_video": ToolSpec(
        name="transcribe_audio_or_video",
        description="Transcribe an audio or video file using Whisper.",
        params={
            "path":             Param("path", "string", True),
            "model_name":       Param("model_name", "string", False, "small", has_default=True,
                                       enum=["tiny", "base", "small", "medium", "large"]),
            "language":         Param("language", "string", False, None),
            "backend":          Param("backend", "string", False, "auto", has_default=True),
            "word_timestamps":  Param("word_timestamps", "boolean", False, False, has_default=True),
        },
    ),
    "transcribe_multiple": ToolSpec(
        name="transcribe_multiple",
        description="Transcribe multiple audio/video files in parallel.",
        params={
            "paths":          Param("paths", "array", True),
            "model_name":     Param("model_name", "string", False, "small", has_default=True),
            "max_concurrent": Param("max_concurrent", "integer", False, 2, has_default=True),
        },
    ),
    # ── Pipeline (the `service` argument is injected by the runtime, not
    #    exposed to the LLM) ─────────────────────────────────────────────
    "process_file_for_vector_store": ToolSpec(
        name="process_file_for_vector_store",
        description="Extract, chunk, embed, and index a file into the vector store.",
        params={
            "file_path":  Param("file_path", "string", True),
            "chunk_size": Param("chunk_size", "integer", False, 1000, has_default=True),
        },
    ),
    "transcribe_url_to_vectorstore": ToolSpec(
        name="transcribe_url_to_vectorstore",
        description="Download a URL, transcribe/extract, and index into the vector store.",
        params={
            "url":          Param("url", "string", True),
            "source_type":  Param("source_type", "string", False, "web", has_default=True),
            "model_name":   Param("model_name", "string", False, "small", has_default=True),
            "persist_db":   Param("persist_db", "boolean", False, False, has_default=True),
        },
    ),
    "transcribe_local_to_vectorstore": ToolSpec(
        name="transcribe_local_to_vectorstore",
        description="Extract a local file, chunk, embed, and index into the vector store.",
        params={
            "path":         Param("path", "string", True),
            "source_type":  Param("source_type", "string", False, "local", has_default=True),
            "model_name":   Param("model_name", "string", False, "small", has_default=True),
            "persist_db":   Param("persist_db", "boolean", False, False, has_default=True),
        },
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Text hygiene — P1
# ═════════════════════════════════════════════════════════════════════════════
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN_TAIL = re.compile(r"<think>.*\Z", re.S | re.I)
_THINK_ORPHAN_CLOSE = re.compile(r"</think>", re.I)


def strip_think(text: str) -> str:
    """
    Remove <think>…</think> blocks from a model reply, plus orphan tags.

    Handles three cases:
      1. Closed block: `<think>…</think>` — removed
      2. Orphan closer: `…</think>answer` — everything up to and including
         the last `</think>` is dropped (the opening tag was in the prompt)
      3. Unclosed generation: `<think>cut off mid` — everything from `<think>`
         to the end is dropped
    """
    if not text:
        return ""
    t = _THINK_BLOCK.sub("", text)
    if _THINK_ORPHAN_CLOSE.search(t):
        t = _THINK_ORPHAN_CLOSE.split(t)[-1]
    t = _THINK_OPEN_TAIL.sub("", t)
    return t.strip()


# ═════════════════════════════════════════════════════════════════════════════
# JSON object scanner — P2
# ═════════════════════════════════════════════════════════════════════════════
def iter_json_objects(text: str) -> Iterator[Tuple[int, int, Any]]:
    """
    Yield (start_index, end_index_exclusive, parsed_object) for every top-level
    `{...}` in `text` that parses as valid JSON.

    String-aware, escape-aware, any nesting depth. Does not use regex, so it
    is not susceptible to catastrophic backtracking. Linear in the size of
    the input.
    """
    if not text:
        return
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0 and j < n:
            chunk = text[i:j + 1]
            try:
                yield i, j + 1, json.loads(chunk)
                i = j + 1
                continue
            except json.JSONDecodeError:
                pass
        i += 1


# ═════════════════════════════════════════════════════════════════════════════
# Tool-name resolution — P4
# ═════════════════════════════════════════════════════════════════════════════
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_tool_name(raw: str, tools: Dict[str, ToolSpec]) -> Optional[str]:
    """
    Match `raw` against the tool names, in priority order:
      1. Exact match
      2. Normalised match (strip punctuation/case)
      3. Alias match
      4. Unique close match — only if clearly better than the runner-up

    Returns the canonical name or None. Never raises. Never fuzzy-matches on
    arbitrary strings (a tool mentioned in prose is not executed).
    """
    if not raw or not isinstance(raw, str):
        return None
    if raw in tools:
        return raw
    n = _norm(raw)
    if not n:
        return None
    by_norm: Dict[str, str] = {_norm(k): k for k in tools}
    if n in by_norm:
        return by_norm[n]
    for k, t in tools.items():
        if n in {_norm(a) for a in t.aliases}:
            return k
    close = difflib.get_close_matches(n, list(by_norm), n=2, cutoff=0.88)
    if not close:
        return None
    if len(close) == 1:
        return by_norm[close[0]]
    ratio_0 = difflib.SequenceMatcher(None, n, close[0]).ratio()
    ratio_1 = difflib.SequenceMatcher(None, n, close[1]).ratio()
    if ratio_0 - ratio_1 > 0.05:
        return by_norm[close[0]]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Handle resolution
# ═════════════════════════════════════════════════════════════════════════════
_HANDLE_RE = re.compile(r"^\$(r\d+)((?:\.[\w-]+)*)$")


def _resolve_handles(value: Any, resolver: Optional[Callable[[str], Any]]) -> Any:
    """Recursively replace $rN.path.to.value handles in the argument tree."""
    if resolver is None:
        return value
    if isinstance(value, str):
        s = value.strip()
        if _HANDLE_RE.match(s):
            try:
                return resolver(s)
            except Exception:
                return value
        return value
    if isinstance(value, list):
        return [_resolve_handles(v, resolver) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_handles(v, resolver) for v in value)
    if isinstance(value, dict):
        return {k: _resolve_handles(v, resolver) for k, v in value.items()}
    return value


# ═════════════════════════════════════════════════════════════════════════════
# Coercion — P5
# ═════════════════════════════════════════════════════════════════════════════
_TRUE_STRINGS = frozenset({"true", "yes", "y", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "no", "n", "0", "off", ""})


def _coerce(v: Any, p: Param) -> Any:
    """
    Coerce `v` to the JSON type of `p`. Only coerces when the target type is
    clearly different from the input. Never converts `"12345"` to int when the
    parameter is a string.
    """
    try:
        if p.type == "integer":
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                try:
                    return int(float(s))
                except (ValueError, TypeError):
                    return v
        if p.type == "number":
            if isinstance(v, bool):
                return float(v)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except (ValueError, TypeError):
                    return v
        if p.type == "boolean":
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in _TRUE_STRINGS:
                    return True
                if s in _FALSE_STRINGS:
                    return False
                return v
        if p.type == "array":
            if isinstance(v, list):
                return v
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("["):
                    try:
                        parsed = json.loads(s)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except json.JSONDecodeError:
                        pass
                # Comma-split only if the string has multiple entries.
                if "," in s:
                    return [x.strip() for x in s.split(",") if x.strip()]
                return [s] if s else []
            if v is None:
                return []
            return [v]
        if p.type == "object":
            if isinstance(v, dict):
                return v
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    return v
            return v
        if p.type == "null":
            return None
        if p.type == "string":
            if isinstance(v, str):
                return v
            if v is None:
                return ""
            if isinstance(v, (list, tuple)):
                return ", ".join(str(x) for x in v)
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return str(v)
    except Exception:
        return v
    return v


# ═════════════════════════════════════════════════════════════════════════════
# Action
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Action:
    """
    One parsed action from the model.

    kind = "tool"   — call the named tool with the given args
    kind = "final"  — the model declared the step done
    kind = "none"   — nothing valid was parsed

    `errors` are human-readable rejection reasons. `notes` are non-fatal
    observations ("tool name normalised", "dropped unknown argument"). An
    Action is only `ok` when `kind` is tool/final AND `errors` is empty.
    """
    kind: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind in ("tool", "final") and not self.errors

    def repair_hint(self, tools: Dict[str, ToolSpec]) -> str:
        """
        One-line hint to send back to the model in a repair turn.
        Shows the exact shape expected.
        """
        if self.kind == "none":
            return (
                'Reply with ONE JSON object only: '
                '{"tool": "<name>", "args": {...}} or '
                '{"tool": null, "final_answer": "..."}'
            )
        spec = tools.get(self.tool or "") if self.tool else None
        msg = "; ".join(self.errors) if self.errors else "malformed action"
        if spec is not None:
            return f"{msg}. Correct call looks like: {spec.example()}"
        return msg


# ═════════════════════════════════════════════════════════════════════════════
# Candidate extraction — P7
# ═════════════════════════════════════════════════════════════════════════════
_ENVELOPE_KEYS = frozenset({
    "tool", "action", "name", "function",
    "args", "arguments", "parameters",
    "final_answer", "answer",
    "thought", "reasoning", "done", "result", "confidence",
})


def _candidate(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Normalise the many shapes models emit into {tool, args, lifted, final},
    or return None if the object is not action-like.
    """
    if not isinstance(obj, dict):
        return None

    # OpenAI tool_calls
    if isinstance(obj.get("tool_calls"), list) and obj["tool_calls"]:
        obj = obj["tool_calls"][0]

    # Wrapped function object
    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        merged = {k: v for k, v in obj.items() if k != "function"}
        merged.update(fn)
        obj = merged

    has_tool = any(k in obj for k in ("tool", "action", "name"))
    has_final = (
        any(k in obj for k in ("final_answer", "answer"))
        or obj.get("done") is True
    )
    if not (has_tool or has_final):
        return None

    tool = obj.get("tool", obj.get("action", obj.get("name")))
    args = obj.get("args", obj.get("arguments", obj.get("parameters")))

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    lifted: Dict[str, Any] = {}
    if args is None:
        args = {}
        lifted = {k: v for k, v in obj.items() if k not in _ENVELOPE_KEYS}

    return {
        "tool": tool,
        "args": args if isinstance(args, dict) else {},
        "lifted": lifted,
        "final": obj.get("final_answer", obj.get("answer")),
    }


# Legacy XML
_XML_TOOL = re.compile(r"<tool>\s*(.+?)\s*</tool>", re.S | re.I)
_XML_ARG = re.compile(r"<arg\s+name=[\"'](\w+)[\"']\s*>(.*?)</arg>", re.S | re.I)


# ═════════════════════════════════════════════════════════════════════════════
# parse_action — the main entry point
# ═════════════════════════════════════════════════════════════════════════════
def parse_action(
    text: str,
    tools: Dict[str, ToolSpec],
    *,
    resolver: Optional[Callable[[str], Any]] = None,
    inject_defaults: bool = False,
    strict_unknown_args: bool = False,
) -> Action:
    """
    Parse a model reply into a validated Action.

    Parameters
    ----------
    text               Raw model output.
    tools              The registry of known tools.
    resolver           Optional callable that resolves $rN.path handles.
                       Called on any string matching `^\\$r\\d+(\\.\\w+)*$`.
    inject_defaults    If True, non-None defaults for optional parameters the
                       model omitted are filled in. Off by default to keep
                       parse results minimal and predictable.
    strict_unknown_args
                       If True, unknown arguments are rejected instead of
                       being silently dropped.

    Returns
    -------
    Action with kind in {"tool", "final", "none"}. Invalid tool actions have
    their reason in `.errors` and a repair hint available via `.repair_hint()`.
    """
    visible = strip_think(text or "")

    # 1. JSON scan — P2, P3, P7
    candidates: List[Dict[str, Any]] = []
    for _, _, obj in iter_json_objects(visible):
        c = _candidate(obj)
        if c is not None:
            candidates.append(c)

    # 2. XML fallback — P7
    if not candidates:
        m = _XML_TOOL.search(visible)
        if m:
            xml_args = {k: v.strip() for k, v in _XML_ARG.findall(visible)}
            candidates = [{
                "tool": m.group(1).strip(),
                "args": xml_args,
                "lifted": {},
                "final": None,
            }]

    if not candidates:
        return Action("none", errors=["no JSON action found"])

    # 3. Pick the last candidate that references a known tool, or the last
    #    candidate that is a valid final answer. This is the P3 rule.
    def _usable(c: Dict[str, Any]) -> bool:
        t = c.get("tool")
        if t in (None, "", "null", "none", "None"):
            return c.get("final") is not None
        if not isinstance(t, str):
            return False
        return resolve_tool_name(t, tools) is not None

    chosen = next((c for c in reversed(candidates) if _usable(c)), candidates[-1])

    # 4. Final answer path
    t_raw = chosen.get("tool")
    if t_raw in (None, "", "null", "none", "None"):
        if chosen.get("final") is not None:
            return Action("final", final_answer=str(chosen["final"]))
        return Action(
            "tool", tool=None,
            errors=["missing 'tool' field"],
        )

    # 5. Resolve tool name — P4
    name = resolve_tool_name(str(t_raw), tools)
    if not name:
        near = difflib.get_close_matches(str(t_raw), list(tools), n=3, cutoff=0.5)
        hint = f" (did you mean {', '.join(near)}?)" if near else ""
        return Action(
            "tool", tool=str(t_raw),
            errors=[f"unknown tool '{t_raw}'{hint}"],
        )
    spec = tools[name]

    # 6. Build args
    act = Action("tool", tool=name)
    if name != t_raw:
        act.notes.append(f"tool name '{t_raw}' normalised to '{name}'")

    raw: Dict[str, Any] = dict(chosen.get("args") or {})
    for k, v in (chosen.get("lifted") or {}).items():
        if k in spec.params and k not in raw:
            raw[k] = v
            act.notes.append(f"lifted root-level '{k}' into args")

    for k, v in raw.items():
        # Handle resolution BEFORE coercion — the resolver returns the real
        # value; coercion then normalises it to the declared type.
        v = _resolve_handles(v, resolver)
        if k in spec.params:
            act.args[k] = _coerce(v, spec.params[k])
        elif spec.accepts_extra:
            act.args[k] = v
        elif strict_unknown_args:
            act.errors.append(f"unknown argument '{k}'")
        else:
            act.notes.append(f"dropped unknown argument '{k}'")

    # 7. Default injection (optional)
    if inject_defaults:
        for p in spec.params.values():
            if p.name in act.args:
                continue
            if not p.required and p.has_default and p.default is not None:
                act.args[p.name] = p.default

    # 8. Required-parameter check — P6
    missing: List[str] = []
    for p in spec.params.values():
        if not p.required:
            continue
        if p.name not in act.args or act.args[p.name] in (None, ""):
            missing.append(p.name)
    if missing:
        act.errors.append(f"missing required argument(s): {', '.join(missing)}")

    # 9. Placeholder guard — small models copy schema examples literally
    #    ("value", "https://example.org/page", "kipling+2026" from the
    #    template). Reject before execution so the repair turn costs one
    #    cheap call instead of a failed tool round.
    _PLACEHOLDERS = {
        "value", "string", "example", "your_value", "<value>",
        "foo", "bar", "baz",
        "https://example.com", "https://example.org",
        "https://example.org/page", "http://example.com",
        "path/to/file", "your_query", "query_here",
    }
    bad = [
        k for k, v in act.args.items()
        if isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS
    ]
    if bad:
        act.errors.append(
            "argument(s) contain placeholder text, use real values from "
            "the task or KNOWN FACTS: " + ", ".join(bad)
        )

    return act

# ═════════════════════════════════════════════════════════════════════════════
# parse_choice — forced-choice mode for tiny models
# ═════════════════════════════════════════════════════════════════════════════
def parse_choice(text: str, n: int) -> Optional[int]:
    """
    Forced-choice mode: the prompt lists `1) tool_a 2) tool_b …` and the model
    answers with a digit. Returns the zero-based index, or None.
    """
    if not text or n <= 0:
        return None
    cleaned = strip_think(text)
    m = re.search(r"\b(\d{1,2})\b", cleaned)
    if not m:
        return None
    k = int(m.group(1)) - 1
    return k if 0 <= k < n else None


# ═════════════════════════════════════════════════════════════════════════════
# Tool-card compilation
# ═════════════════════════════════════════════════════════════════════════════
def compile_tool_cards(
    specs: Dict[str, ToolSpec],
    names: List[str],
    *,
    budget_chars: int = 900,
    numbered: bool = False,
    include_example: int = 0,
    include_description: bool = True,
) -> str:
    """
    Build a compact, char-budgeted tool listing for a single prompt.

    Format:
        - name(req*, opt:int=default) — one-line description
        - name(req*, opt:int=default) — one-line description
        ... (N more)

    Parameters
    ----------
    specs              Full spec map.
    names              Ordered tool names to consider.
    budget_chars       Hard cap on the rendered block (excluding examples).
    numbered           Prefix each line with "1) " instead of "- " (forced
                       choice).
    include_example    Append up to this many `ToolSpec.example()` lines
                       (zero to skip).
    include_description
                       When False, emit only signatures — for the very small
                       budget of a nano-tier model.

    Tools past the budget are dropped and summarised in one "... (N more)"
    line so the model knows they exist and the runtime can surface them on a
    repair turn.
    """
    if not names:
        return ""
    lines: List[str] = []
    used = 0
    shown = 0

    for i, n in enumerate(names):
        spec = specs.get(n)
        if spec is None:
            card = n
        else:
            card = spec.card() if include_description else spec.card().split(" — ")[0]
        prefix = f"{i + 1}) " if numbered else "- "
        line = prefix + card
        # Reserve room for a possible "... (N more)" trailer on the last line
        remaining_after = len(names) - (shown + 1)
        reserve = len(f"\n... ({remaining_after} more)") if remaining_after > 0 else 0
        if used + len(line) + reserve + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1

    if shown < len(names):
        lines.append(f"... ({len(names) - shown} more)")

    if include_example > 0 and shown > 0:
        lines.append("")
        lines.append("Example:")
        for n in names[:include_example]:
            spec = specs.get(n)
            if spec is not None:
                lines.append("  " + spec.example())

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Registry helper — build specs with overrides applied
# ═════════════════════════════════════════════════════════════════════════════
def build_specs(
    callables: Dict[str, Callable],
    *,
    aliases: Optional[Dict[str, Tuple[str, ...]]] = None,
    extra_overrides: Optional[Dict[str, ToolSpec]] = None,
    hide_prefixes: Tuple[str, ...] = (),
) -> Dict[str, ToolSpec]:
    """
    Build a ToolSpec map from {name: callable}, applying overrides and hiding
    factory / internal tools.

    Priority:
      1. `extra_overrides` (caller-supplied)
      2. `TOOL_SPEC_OVERRIDES` (bundled)
      3. `ToolSpec.from_callable` (auto-introspection)

    Parameters
    ----------
    callables        name -> callable.
    aliases          Optional name -> alias tuple.
    extra_overrides  Per-project overrides; wins over bundled ones.
    hide_prefixes    Names starting with any of these are excluded
                     (e.g. ("create_", "check_", "get_default_")).
    """
    aliases = aliases or {}
    extra_overrides = extra_overrides or {}
    specs: Dict[str, ToolSpec] = {}
    for name, fn in callables.items():
        if hide_prefixes and any(name.startswith(p) for p in hide_prefixes):
            continue
        if name in extra_overrides:
            specs[name] = extra_overrides[name]
        elif name in TOOL_SPEC_OVERRIDES:
            specs[name] = TOOL_SPEC_OVERRIDES[name]
        else:
            specs[name] = ToolSpec.from_callable(name, fn, aliases=aliases.get(name, ()))
    return specs


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════
__all__ = [
    # data
    "Param", "ToolSpec", "Action",
    # overrides
    "TOOL_SPEC_OVERRIDES",
    # parsing
    "parse_action", "parse_choice", "strip_think", "iter_json_objects",
    "resolve_tool_name",
    # rendering
    "compile_tool_cards",
    # registry helper
    "build_specs",
]
