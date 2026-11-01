"""
Hugging Face Dataset Explorer — single-file FastAPI app (Vercel-ready).

A thin, cached proxy over the public Hugging Face Dataset Viewer API.
Works with any datasets-server-visible dataset; nothing is hardcoded.

    pip install -r requirements.txt
    export HF_DATASET=stanfordnlp/imdb        # or put it in Vercel env vars
    uvicorn hf_explorer:app --host 0.0.0.0 --port 7860

    http://localhost:7860/            browser UI
    http://localhost:7860/health
    http://localhost:7860/api/dataset
    http://localhost:7860/api/rows?config=plain_text&split=train&offset=0&length=20
    http://localhost:7860/api/search?q=great+film&config=plain_text&split=train

Env vars
    HF_DATASET              dataset id, e.g. "stanfordnlp/imdb"   (required)
    HF_TOKEN                optional; needed for gated/private datasets
    ALLOW_DATASET_OVERRIDE  "1" lets ?dataset= pick a different dataset per request
                            (default off, so a public deploy can't spend your quota)
    DEMO_API_KEY            set to require X-API-Key on /api/* (default: open)

Fixes vs. the version this replaces
  * /splits metadata is cached (it was fetched before every /api/rows call,
    doubling latency and burning rate limit on every page of results).
  * One retry with backoff on 429/502/503.
  * Upstream error bodies are surfaced instead of a generic "returned HTTP 502".
  * /api/search degrades to a clear 501 when the dataset doesn't support it,
    instead of a confusing 502.
  * Rows are normalised, so the UI doesn't break on nested/None values.
  * Optional API key, off by default so the browser UI still works out of the box.

This app is dataset-agnostic and holds no data of its own — it renders whatever
public dataset you point it at. Only point it at datasets you are allowed to
serve.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ── Configuration ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-explorer")

HF_DATASET = os.getenv("HF_DATASET", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
ALLOW_OVERRIDE = os.getenv("ALLOW_DATASET_OVERRIDE", "0").strip() == "1"
API_KEY = os.getenv("DEMO_API_KEY", "").strip()

HF_API = "https://datasets-server.huggingface.co"
TIMEOUT = httpx.Timeout(20.0, connect=8.0)
RETRY_STATUSES = {429, 502, 503, 504}
SPLITS_TTL = 300.0          # seconds
MAX_ROWS_PER_REQUEST = 100  # Hugging Face hard limit
SCAN_FALLBACK_ROWS = int(os.getenv("SCAN_FALLBACK_ROWS", "2000"))
SCAN_CONCURRENCY = 4
BREAKER_TTL = 60.0          # skip a known-broken upstream endpoint this long
PROBE_TIMEOUT = httpx.Timeout(8.0, connect=5.0)


def get_headers() -> dict[str, str]:
    headers = {"User-Agent": "HF-Dataset-Explorer/1.1"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=TIMEOUT, headers=get_headers(), follow_redirects=True,
    )
    app.state.splits_cache: dict[str, tuple[float, dict]] = {}
    app.state.breaker: dict[str, float] = {}   # endpoint -> failure timestamp
    app.state.cache_lock = asyncio.Lock()
    logger.info("dataset=%s token=%s override=%s",
                HF_DATASET or "(unset)", bool(HF_TOKEN), ALLOW_OVERRIDE)
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="Hugging Face Dataset API",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Upstream helper ─────────────────────────────────────────────────────────
def _resolve_dataset(request: Request, override: str | None) -> str:
    if ALLOW_OVERRIDE and override:
        return override.strip()
    if not HF_DATASET:
        raise HTTPException(
            status_code=503,
            detail="Set the HF_DATASET environment variable.",
        )
    return HF_DATASET


def _upstream_detail(response: httpx.Response) -> str:
    """Pull Hugging Face's own error message out of the body, if any."""
    try:
        body = response.json()
        if isinstance(body, dict):
            for key in ("error", "detail", "message"):
                if body.get(key):
                    return str(body[key])[:400]
    except ValueError:
        pass
    return response.text[:400].strip() or "(empty body)"


def breaker_open(state, endpoint: str) -> bool:
    """True if this endpoint failed recently, so we skip straight past it."""
    failed_at = state.breaker.get(endpoint)
    if failed_at is None:
        return False
    if time.monotonic() - failed_at > BREAKER_TTL:
        state.breaker.pop(endpoint, None)
        return False
    return True


def breaker_trip(state, endpoint: str) -> None:
    if endpoint not in state.breaker:
        logger.warning("circuit opened for /%s for %.0fs", endpoint, BREAKER_TTL)
    state.breaker[endpoint] = time.monotonic()


async def hf_get(request: Request, endpoint: str, params: dict,
                 timeout: httpx.Timeout | None = None) -> dict:
    dataset = params.pop("_dataset")
    query = {"dataset": dataset, **params}
    url = f"{HF_API}/{endpoint.lstrip('/')}"

    last_error: str | None = None

    for attempt in range(2):  # one retry on transient upstream failures
        try:
            response = await request.app.state.client.get(
                url, params=query, timeout=timeout or TIMEOUT)

            if response.status_code in RETRY_STATUSES and attempt == 0:
                wait = 0.6 * (attempt + 1)
                logger.warning(
                    "upstream %s for %s, retrying in %.1fs",
                    response.status_code, endpoint, wait,
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code >= 400:
                detail = _upstream_detail(response)
                logger.error("upstream HTTP %s for %s: %s",
                             response.status_code, endpoint, detail)
                status = 404 if response.status_code == 404 else 502
                raise HTTPException(
                    status_code=status,
                    detail=f"Hugging Face returned HTTP {response.status_code}: {detail}",
                )

            data = response.json()

            if not isinstance(data, dict):
                raise HTTPException(502, "Unexpected response from Hugging Face.")

            if data.get("error"):
                last_error = str(data["error"])
                logger.warning("dataset error from %s: %s", endpoint, last_error)
                raise HTTPException(
                    status_code=502,
                    detail=f"Hugging Face could not serve this dataset: {last_error}",
                )

            return data

        except HTTPException:
            raise
        except httpx.TimeoutException:
            logger.exception("upstream timeout on %s", endpoint)
            if attempt == 0:
                continue
            raise HTTPException(504, "Hugging Face request timed out. Try again.")
        except httpx.RequestError as exc:
            logger.exception("upstream network error on %s", endpoint)
            if attempt == 0:
                continue
            raise HTTPException(502, f"Could not connect to Hugging Face: {exc}")
        except ValueError:
            logger.exception("invalid JSON from %s", endpoint)
            raise HTTPException(502, "Hugging Face returned invalid JSON.")

    raise HTTPException(502, last_error or "Hugging Face request failed.")


async def get_splits(request: Request, dataset: str) -> list[dict]:
    """Cached /splits lookup — this used to run on every single request."""
    cache: dict[str, tuple[float, dict]] = request.app.state.splits_cache
    now = time.monotonic()

    hit = cache.get(dataset)
    if hit and now - hit[0] < SPLITS_TTL:
        return hit[1].get("splits", [])

    async with request.app.state.cache_lock:
        hit = cache.get(dataset)  # re-check inside the lock
        if hit and time.monotonic() - hit[0] < SPLITS_TTL:
            return hit[1].get("splits", [])

        result = await hf_get(request, "splits", {"_dataset": dataset})
        cache[dataset] = (time.monotonic(), result)
        return result.get("splits", [])


async def require_split(request: Request, dataset: str, config: str, split: str) -> None:
    available = await get_splits(request, dataset)
    if not available:
        raise HTTPException(
            422,
            "This dataset has no viewer splits. It may be private, gated, "
            "or not converted to parquet yet.",
        )
    if not any(s.get("config") == config and s.get("split") == split
               for s in available):
        pairs = sorted({(s.get("config"), s.get("split")) for s in available})
        raise HTTPException(
            422,
            f"Invalid config/split {config!r}/{split!r}. "
            f"Available: {pairs[:8]}. See /api/dataset.",
        )


# ── Optional API key ────────────────────────────────────────────────────────
async def api_guard(request: Request) -> None:
    """No-op unless DEMO_API_KEY is set."""
    if not API_KEY:
        return
    import hmac
    provided = request.headers.get("X-API-Key") or request.query_params.get("key")
    if not provided or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(
            401,
            "Missing or invalid API key. Send X-API-Key header.",
        )


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset_configured": bool(HF_DATASET),
        "provider": "Hugging Face Dataset Viewer API",
        "auth_required": bool(API_KEY),
        "dataset_override": ALLOW_OVERRIDE,
    }


@app.get("/api/dataset", dependencies=[Depends(api_guard)])
async def dataset_info(
    request: Request,
    dataset: str | None = Query(default=None, max_length=200),
):
    target = _resolve_dataset(request, dataset)
    result = await hf_get(request, "splits", {"_dataset": target})
    splits = result.get("splits", [])

    return {
        "success": True,
        "dataset": target,
        "split_count": len(splits),
        "splits": splits,
        "pending": result.get("pending", []),
        "failed": result.get("failed", []),
    }


@app.get("/api/rows", dependencies=[Depends(api_guard)])
async def dataset_rows(
    request: Request,
    config: str = Query(default="default", min_length=1, max_length=200),
    split: str = Query(default="train", min_length=1, max_length=100),
    offset: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=MAX_ROWS_PER_REQUEST),
    dataset: str | None = Query(default=None, max_length=200),
):
    target = _resolve_dataset(request, dataset)
    await require_split(request, target, config, split)

    result = await hf_get(request, "rows", {
        "_dataset": target,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })

    return {
        "success": True,
        "dataset": target,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
        "total": result.get("num_rows_total"),
        "features": result.get("features", []),
        "rows": result.get("rows", []),
    }


async def _scan_search(
    request: Request, target: str, config: str, split: str,
    q: str, offset: int, length: int,
) -> tuple[list[dict], int, list[dict]]:
    """Last-resort search: page /rows in parallel and match locally.

    Used only when Hugging Face's own /search and /filter endpoints are
    unavailable. Bounded by SCAN_FALLBACK_ROWS so it can't run away.
    """
    needle = q.lower()
    wanted = offset + length
    collected: list[dict] = []
    scanned = 0
    features: list[dict] = []
    sem = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def fetch(off: int) -> dict | Exception:
        async with sem:
            try:
                return await hf_get(request, "rows", {
                    "_dataset": target, "config": config, "split": split,
                    "offset": off, "length": MAX_ROWS_PER_REQUEST,
                })
            except HTTPException as exc:
                return exc

    offsets = list(range(0, SCAN_FALLBACK_ROWS, MAX_ROWS_PER_REQUEST))
    for i in range(0, len(offsets), SCAN_CONCURRENCY):
        batch = offsets[i:i + SCAN_CONCURRENCY]
        results = await asyncio.gather(*(fetch(o) for o in batch))
        for res in results:
            if isinstance(res, Exception):
                continue
            if not features:
                features = res.get("features", [])
            for item in res.get("rows", []):
                scanned += 1
                row = item.get("row") or {}
                if any(needle in str(v).lower()
                       for v in row.values() if isinstance(v, (str, int, float))):
                    collected.append(item)
        if len(collected) >= wanted:
            break

    return collected[offset:offset + length], scanned, features


@app.get("/api/search", dependencies=[Depends(api_guard)])
async def search_dataset(
    request: Request,
    q: str = Query(min_length=1, max_length=200),
    config: str = Query(default="default"),
    split: str = Query(default="train"),
    offset: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=MAX_ROWS_PER_REQUEST),
    dataset: str | None = Query(default=None, max_length=200),
):
    target = _resolve_dataset(request, dataset)
    await require_split(request, target, config, split)

    base = {"success": True, "dataset": target, "config": config,
            "split": split, "query": q, "offset": offset, "length": length}

    state = request.app.state
    primary_error = "skipped (recently failed)"

    # 1) native server-side search
    if not breaker_open(state, "search"):
        try:
            result = await hf_get(request, "search", {
                "_dataset": target, "config": config, "split": split,
                "query": q, "offset": offset, "length": length,
            }, timeout=PROBE_TIMEOUT)
            return {**base, "mode": "search",
                    "total": result.get("num_rows_total"),
                    "features": result.get("features", []),
                    "rows": result.get("rows", [])}
        except HTTPException as exc:
            primary_error = exc.detail
            breaker_trip(state, "search")
            logger.info("native search unavailable (%s), trying /filter", primary_error)

    # 2) server-side SQL filter over string-ish columns
    if not breaker_open(state, "filter"):
      try:
        features = (await hf_get(request, "rows", {
            "_dataset": target, "config": config, "split": split,
            "offset": 0, "length": 1,
        })).get("features", [])
        cols = [f["name"] for f in features
                if f.get("type", {}).get("_type") in ("Value", "ClassLabel")]
        if cols:
            safe = q.replace("'", "''")
            where = " OR ".join(f'"{c}" LIKE \'%{safe}%\'' for c in cols)
            result = await hf_get(request, "filter", {
                "_dataset": target, "config": config, "split": split,
                "where": where, "offset": offset, "length": length,
            }, timeout=PROBE_TIMEOUT)
            return {**base, "mode": "filter",
                    "total": result.get("num_rows_total"),
                    "features": result.get("features", features),
                    "rows": result.get("rows", [])}
      except HTTPException as exc:
        breaker_trip(state, "filter")
        logger.info("filter endpoint unavailable (%s), scanning instead", exc.detail)

    # 3) bounded local scan
    if SCAN_FALLBACK_ROWS <= 0:
        raise HTTPException(
            501,
            "Server-side search is unavailable and SCAN_FALLBACK_ROWS is 0. "
            f"Upstream said: {primary_error}",
        )

    rows, scanned, features = await _scan_search(
        request, target, config, split, q, offset, length)
    return {**base, "mode": "scan", "scanned_rows": scanned,
            "total": None if scanned < SCAN_FALLBACK_ROWS else None,
            "features": features, "rows": rows,
            "note": (f"Hugging Face search was unavailable; matched locally "
                     f"across the first {scanned} rows of this split.")}


@app.get("/api/parquet")
async def parquet_files(
    request: Request,
    dataset: str | None = Query(default=None, max_length=200),
):
    """Where the actual parquet shards live — handy for bulk downloads."""
    target = _resolve_dataset(request, dataset)
    result = await hf_get(request, "parquet", {"_dataset": target})
    return {"success": True, "dataset": target,
            "parquet_files": result.get("parquet_files", [])}


# ── Browser UI ──────────────────────────────────────────────────────────────
HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HF Dataset Explorer</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        margin: 0 auto; padding: 28px 16px; max-width: 1100px;
        background: #f5f6fa; color: #202536; }
 h1 { margin: 0 0 4px; font-size: 1.4rem; }
 .sub { color: #6b7280; margin: 0 0 20px; font-size: .9rem; }
 section { background: #fff; padding: 16px 18px; margin: 14px 0; border-radius: 12px;
           box-shadow: 0 1px 2px rgba(0,0,0,.06); }
 button, select, input { padding: 9px 11px; margin: 4px 4px 4px 0; font-size: .95rem;
                         border: 1px solid #d3d8e0; border-radius: 8px; background: #fff; }
 button { cursor: pointer; }
 button:disabled { opacity: .45; cursor: default; }
 #table { overflow-x: auto; margin-top: 12px; }
 table { border-collapse: collapse; width: 100%; font-size: .9rem; }
 th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #e5e8ee;
          overflow-wrap: anywhere; vertical-align: top; }
 th { background: #edf0f6; position: sticky; top: 0; }
 .msg { color: #6b7280; }
 .err { color: #b42318; }
 #pager { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
 @media (prefers-color-scheme: dark) {
   body { background: #14161a; color: #e6e8ec; }
   section { background: #1c1f25; box-shadow: none; }
   button, select, input { background: #23272f; color: #e6e8ec; border-color: #333844; }
   th { background: #23272f; } th, td { border-color: #2b3038; }
 }
</style>
</head>
<body>
<h1>Hugging Face Dataset Explorer</h1>
<p class="sub" id="dataset-line">Loading dataset metadata…</p>

<section>
  <p id="status" class="msg">Connecting…</p>
  <label>Config <select id="config"></select></label>
  <label>Split <select id="split"></select></label>
  <button id="load">Load records</button>
</section>

<section>
  <input id="query" placeholder="Search dataset text" style="min-width:260px">
  <button id="search">Search</button>
  <div id="table" class="msg">No data loaded.</div>
  <div id="pager">
    <button id="prev" disabled>← Previous</button>
    <span id="page" class="msg">Page 1</span>
    <button id="next" disabled>Next →</button>
  </div>
</section>

<script>
const $ = id => document.getElementById(id);
const PAGE = 20;
let offset = 0, total = null, splits = [], lastMode = "rows", lastQuery = "";

async function api(path) {
  const response = await fetch(path);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    throw new Error(data.detail || ("HTTP " + response.status));
  }
  return data;
}

const selected = () => JSON.parse($("split").value);

function cellValue(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

function displayRows(rows) {
  if (!rows || !rows.length) {
    $("table").textContent = "No records found.";
    $("table").className = "msg";
    return;
  }
  $("table").className = "";
  const columns = Object.keys(rows[0].row || {});
  const table = document.createElement("table");

  const thead = document.createElement("thead");
  const head = document.createElement("tr");
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col;
    head.appendChild(th);
  }
  thead.appendChild(head);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const item of rows) {
    const tr = document.createElement("tr");
    for (const col of columns) {
      const td = document.createElement("td");
      td.textContent = cellValue((item.row || {})[col]);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  $("table").replaceChildren(table);
}

function setPager() {
  const pageNo = Math.floor(offset / PAGE) + 1;
  $("page").textContent = "Page " + pageNo +
    (total !== null ? " · " + total.toLocaleString() + " rows" : "");
  $("prev").disabled = offset === 0;
  $("next").disabled = total !== null && offset + PAGE >= total;
}

async function loadRows() {
  lastMode = "rows";
  $("status").textContent = "Loading records…";
  $("status").className = "msg";
  $("prev").disabled = true; $("next").disabled = true;
  try {
    const s = selected();
    const params = new URLSearchParams({
      config: s.config, split: s.split,
      offset: String(offset), length: String(PAGE),
    });
    const data = await api("/api/rows?" + params);
    displayRows(data.rows || []);
    total = data.total ?? null;
    $("status").textContent = "Loaded " + (data.rows || []).length + " records.";
    setPager();
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
    $("status").className = "err";
    $("table").textContent = "Could not load records.";
    $("table").className = "msg";
  }
}

async function searchRows() {
  const q = $("query").value.trim();
  if (!q) { $("status").textContent = "Enter a search term."; return; }
  lastMode = "search"; lastQuery = q;
  $("status").textContent = "Searching…";
  $("status").className = "msg";
  try {
    const s = selected();
    const params = new URLSearchParams({ q, config: s.config, split: s.split,
                                         offset: String(offset), length: String(PAGE) });
    const data = await api("/api/search?" + params);
    displayRows(data.rows || []);
    total = data.total ?? null;
    $("status").textContent = "Search: " + (data.rows || []).length + " matches.";
    setPager();
  } catch (e) {
    $("status").textContent = "Search error: " + e.message;
    $("status").className = "err";
  }
}

async function init() {
  try {
    const data = await api("/api/dataset");
    splits = data.splits || [];
    $("dataset-line").textContent = "Dataset: " + data.dataset +
      " · " + splits.length + " split(s)";
    if (!splits.length) throw new Error("No available dataset splits.");

    const configs = [...new Set(splits.map(x => x.config))];
    $("config").replaceChildren();
    for (const config of configs) {
      const opt = document.createElement("option");
      opt.value = config; opt.textContent = config;
      $("config").appendChild(opt);
    }

    function updateSplits() {
      const config = $("config").value;
      $("split").replaceChildren();
      for (const item of splits.filter(x => x.config === config)) {
        const opt = document.createElement("option");
        opt.value = JSON.stringify(item);
        opt.textContent = item.split;
        $("split").appendChild(opt);
      }
      offset = 0; total = null;
    }

    $("config").addEventListener("change", () => { updateSplits(); loadRows(); });
    $("split").addEventListener("change", () => { offset = 0; total = null; loadRows(); });
    updateSplits();
    await loadRows();
  } catch (e) {
    $("dataset-line").textContent = "Dataset error";
    $("status").textContent = "Dataset error: " + e.message;
    $("status").className = "err";
  }
}

$("load").onclick = () => { offset = 0; loadRows(); };
$("search").onclick = () => { offset = 0; searchRows(); };
$("query").addEventListener("keydown", e => {
  if (e.key === "Enter") { offset = 0; searchRows(); }
});
$("prev").onclick = () => {
  offset = Math.max(0, offset - PAGE);
  lastMode === "search" ? searchRows() : loadRows();
};
$("next").onclick = () => {
  if (total !== null && offset + PAGE >= total) return;
  offset += PAGE;
  lastMode === "search" ? searchRows() : loadRows();
};

init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HOME_HTML)


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    """JSON errors everywhere, so the UI's fetch() never chokes on HTML."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail},
    )


# Vercel Python runtime entry point: the ASGI object is `app` above.
