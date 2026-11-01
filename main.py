"""
================================================================================
 DATASET JSON API  —  single file (Vercel Python runtime ready)
================================================================================
 Har endpoint ka kaam ALAG hai (response shape bhi alag):

  1. GET  /                        -> HTML docs + live tester
  2. GET  /health                  -> health JSON
  3. GET  /api/dataset             -> configs / splits / size
  4. GET  /api/fields              -> schema (column + dtype)
  5. GET  /api/key/{key}           -> SIRF key check (koi data nahi)
  6. GET  /{key}                   -> BROWSE data  (page based)   JSON
  7. GET  /{key}/{query}           -> SEARCH data  (matches meta)  JSON
  8. GET  /api/v1/{key}            -> BROWSE data  (cursor based)
  9. GET  /api/v1/{key}/{query}    -> SEARCH data  (+ field_hits)

 Sabhi data routes par:
   ?ui=1 / browser  -> infinite-scroll HTML UI (har record pretty JSON)
   ?format=json     -> pretty-printed JSON (indent 2)
   ?format=compact  -> compact JSON
   ?limit=100       -> per request kitne rows (HF max 100)
   ?page=2          -> agla page  (infinite scroll yahi karta hai)
   ?notify=0        -> bot notification off

 Env vars (.env locally, Vercel -> Settings -> Environment Variables):
   HF_DATASET, HF_TOKEN, HF_CONFIG, HF_SPLIT,
   ALLOWED_KEYS (comma-separated; khaali / * = sab allowed),
   PAGE_SIZE, MAX_PAGE_SIZE, CACHE_TTL, SCAN_MAX_ROWS,
   BOT_TOKEN, BOT_CHAT_ID, BOT_SILENT, API_TITLE, LOCATION_LOOKUP
================================================================================
"""

import os
import re
import json
import time
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ------------------------------------------------------------------ bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset-json-api")


def load_dotenv(path: str = ".env") -> None:
    """Chhota .env loader — koi extra dependency nahi. Pehle se set var override nahi hota."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        target = next((p for p in (path, os.path.join(here, path)) if os.path.isfile(p)), None)
        if not target:
            return
        with open(target, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if name and name not in os.environ:
                    os.environ[name] = value
    except Exception:
        logger.warning("Could not read .env (ignored)")


load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else str(value).strip()


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_env(name, str(default)) or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


HF_DATASET = _env("HF_DATASET")
HF_TOKEN = _env("HF_TOKEN")
HF_CONFIG = _env("HF_CONFIG", "default") or "default"
HF_SPLIT = _env("HF_SPLIT", "train") or "train"

HF_API = "https://datasets-server.huggingface.co"
HF_MAX_LIMIT = 100
TIMEOUT = httpx.Timeout(30.0, connect=8.0)

PAGE_SIZE = _env_int("PAGE_SIZE", 20, 1, HF_MAX_LIMIT)
MAX_PAGE_SIZE = _env_int("MAX_PAGE_SIZE", 100, 1, HF_MAX_LIMIT)
CACHE_TTL = _env_int("CACHE_TTL", 120, 0, 86400)
SCAN_MAX_ROWS = _env_int("SCAN_MAX_ROWS", 3000, 100, 500000)
# Vercel par function 10s (Hobby) me band ho jata hai -> budget ke andar raho
REQUEST_BUDGET = float(_env("REQUEST_BUDGET", "7.5") or 7.5)
HF_CALL_TIMEOUT = float(_env("HF_CALL_TIMEOUT", "6") or 6)
SEARCH_CACHE_TTL = _env_int("SEARCH_CACHE_TTL", 300, 0, 86400)
CLIENT_MAX_RETRIES = _env_int("CLIENT_MAX_RETRIES", 3, 0, 10)

BOT_TOKEN = _env("BOT_TOKEN") or _env("TELEGRAM_BOT_TOKEN")
BOT_CHAT_ID = _env("BOT_CHAT_ID") or _env("TELEGRAM_CHAT_ID")
BOT_SILENT = _env("BOT_SILENT", "0") == "1"
API_TITLE = _env("API_TITLE", "my Student Data API") or "my Student Data API"
LOCATION_LOOKUP = _env("LOCATION_LOOKUP", "1") == "1"
BOT_ENABLED = bool(BOT_TOKEN and BOT_CHAT_ID)


# ------------------------------------------------------------- allowed keys --

def parse_keys(raw: str) -> list:
    """
    ALLOWED_KEYS ko jitne bhi shakal me daala ho, sab chalta hai:
      arya,key2            -> 2 keys
      ["arya", "key2"]     -> 2 keys   (JSON list / python list style)
      'arya' ; key2        -> 2 keys
      arya key2            -> 2 keys
      "arya",              -> 1 key
    Brackets, quotes, spaces, newlines sab strip ho jate hain.
    """
    text = (raw or "").strip()
    if not text:
        return []

    # JSON array ho to seedha parse
    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                items = []
                for item in data:
                    items.extend(parse_keys(str(item)))
                return items
        except Exception:
            text = text.strip("[]")

    text = text.strip("[](){}")
    items = re.split(r"[,;\n\r\t ]+", text)
    cleaned = []
    for item in items:
        item = item.strip().strip("'\"").strip().rstrip(",").strip()
        if item:
            cleaned.append(item)
    return cleaned


_RAW_KEYS = _env("ALLOWED_KEYS")
ALLOWED_KEYS = parse_keys(_RAW_KEYS)
# '*' ya 'all' = sab keys allowed
ALLOW_ALL_KEYS = (not ALLOWED_KEYS) or any(k in ("*", "all", "any") for k in ALLOWED_KEYS)
ALLOWED_KEY_SET = {k.casefold() for k in ALLOWED_KEYS if k not in ("*", "all", "any")}


def mask_key(key: str, keep: int = 2) -> str:
    """arya -> ar*a · 14ec092a -> 14****2a · a -> *"""
    if not key:
        return "****"
    if len(key) <= 2:
        return "*" * max(2, len(key))
    if len(key) <= keep * 2:
        return f"{key[0]}{'*' * max(1, len(key) - 2)}{key[-1]}"
    return f"{key[:keep]}{'*' * min(8, len(key) - keep * 2)}{key[-keep:]}"


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def check_key(key: str) -> dict:
    """Lenient check: trim + case-insensitive + quotes ignore. Detail me debug info bhi."""
    raw = key or ""
    key = raw.strip().strip("'\"").strip()

    if not key or len(key) > 128:
        raise HTTPException(status_code=400, detail="Invalid key: empty or too long.")
    if not re.fullmatch(r"[A-Za-z0-9_.:@-]+", key):
        raise HTTPException(
            status_code=400,
            detail="Invalid key format. Allowed characters: A-Z a-z 0-9 _ . : @ -",
        )

    if ALLOW_ALL_KEYS:
        return {"key": key, "restricted": False}

    if key.casefold() in ALLOWED_KEY_SET:
        return {"key": key, "restricted": True}

    masked = ", ".join(mask_key(k) for k in ALLOWED_KEYS)
    raise HTTPException(
        status_code=403,
        detail=(
            f"This key is not allowed. Tried: {mask_key(key)} "
            f"(id {key_id(key)}). Configured keys ({len(ALLOWED_KEYS)}): {masked}. "
            "Tip: ALLOWED_KEYS me sirf keys rakho (comma separated), ya khaali chhodo / '*' "
            "daalo to har key chalne lage."
        ),
    )


# ------------------------------------------------------------------ helpers --

_CACHE: dict[str, tuple[float, Any]] = {}
_GEO_CACHE: dict[str, tuple[float, dict]] = {}


def cache_get(key: str):
    item = _CACHE.get(key)
    if not item or not CACHE_TTL:
        return None
    stamp, value = item
    if (time.time() - stamp) > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    if CACHE_TTL:
        _CACHE[key] = (time.time(), value)


def get_headers() -> dict:
    headers = {"User-Agent": "Dataset-JSON-API/3.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return str(value)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(clean(value), ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def pretty(payload: dict, compact: bool = False) -> Response:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        default=str,
    )
    return Response(content=body, media_type="application/json; charset=utf-8")


# ----------------------------------------------------------------- lifespan --

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=TIMEOUT,
        headers=get_headers(),
        follow_redirects=True,
    )
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="Dataset JSON API",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------- HF transport --

async def hf_get(endpoint: str, params: dict, call_timeout: Optional[float] = None) -> dict:
    if not HF_DATASET:
        raise HTTPException(status_code=503, detail="Set the HF_DATASET environment variable.")

    try:
        response = await app.state.client.get(
            f"{HF_API}/{endpoint.lstrip('/')}",
            params={"dataset": HF_DATASET, **params},
            timeout=httpx.Timeout(
                call_timeout or HF_CALL_TIMEOUT,
                connect=min(6.0, (call_timeout or HF_CALL_TIMEOUT)),
            ),
        )
    except httpx.TimeoutException:
        logger.exception("HF timeout (%s)", endpoint)
        raise HTTPException(status_code=504, detail="Hugging Face request timed out. Retry.")
    except httpx.RequestError:
        logger.exception("HF network error (%s)", endpoint)
        raise HTTPException(status_code=502, detail="Could not connect to Hugging Face.")

    if response.status_code >= 400:
        logger.error("HF HTTP %s on %s :: %s", response.status_code, endpoint, response.text[:300])
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API returned HTTP {response.status_code}.",
        )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Hugging Face returned invalid JSON.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected response from Hugging Face.")

    if data.get("error"):
        logger.warning("HF dataset error: %s", data["error"])
        raise HTTPException(status_code=502, detail="Hugging Face could not serve this dataset.")

    return data


async def get_splits() -> list:
    cached = cache_get("splits")
    if cached is not None:
        return cached
    result = await hf_get("splits", {})
    splits = result.get("splits", []) or []
    cache_set("splits", splits)
    return splits


async def resolve_split(config: str, split: str) -> dict:
    splits = await get_splits()
    for item in splits:
        if item.get("config") == config and item.get("split") == split:
            return item
    if config == HF_CONFIG and split == HF_SPLIT:
        for item in splits:
            if item.get("split") == HF_SPLIT:
                return item
        for item in splits:
            if item.get("split") in ("train", "test", "validation"):
                return item
        if splits:
            return splits[0]
    raise HTTPException(
        status_code=422,
        detail="Invalid config or split. See /api/dataset for available values.",
    )


async def fetch_rows(config: str, split: str, offset: int, length: int,
                     call_timeout: Optional[float] = None) -> dict:
    return await hf_get(
        "rows",
        {
            "config": config,
            "split": split,
            "offset": max(0, offset),
            "length": max(1, min(HF_MAX_LIMIT, length)),
        },
        call_timeout=call_timeout,
    )


async def get_schema(config: str, split: str) -> list:
    cache_key = f"schema::{config}::{split}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    data = await fetch_rows(config, split, 0, 1, call_timeout=min(HF_CALL_TIMEOUT, 4.0))
    features = data.get("features", []) or []
    cache_set(cache_key, features)
    cache_set(f"total::{config}::{split}", int(data.get("num_rows_total") or 0))
    return features


def features_to_schema(features: list) -> list:
    out = []
    for feature in features or []:
        name = feature.get("name")
        if not name:
            continue
        dtype = ((feature.get("type") or {}).get("dtype") or (feature.get("type") or {}).get("_type") or "string")
        out.append({"name": name, "dtype": str(dtype).lower()})
    return out


def normalise_records(rows: list, start_index: Optional[int] = None) -> list:
    out = []
    for i, item in enumerate(rows or []):
        row = item.get("row") if isinstance(item, dict) else None
        if not isinstance(row, dict):
            row = item if isinstance(item, dict) else {"value": item}
        record = clean(row)
        idx = item.get("row_idx") if isinstance(item, dict) else None
        if idx is None and start_index is not None:
            idx = start_index + i
        if idx is not None:
            record["_index"] = idx
        out.append(record)
    return out


# ------------------------------------------------------------------ search --

def build_where(features: list, query: str) -> Optional[str]:
    q = query.replace("'", "''")
    parts = []
    for feature in features or []:
        name = feature.get("name")
        if not name or any(ch in name for ch in ('"', "\\", "\n")):
            continue
        parts.append(f'CAST("{name}" AS VARCHAR) ILIKE \'%{q}%\'')
    return " OR ".join(parts) if parts else None


async def hf_filter(config: str, split: str, where: str, offset: int, length: int) -> dict:
    return await hf_get(
        "filter",
        {
            "config": config,
            "split": split,
            "where": where,
            "offset": offset,
            "length": max(1, min(HF_MAX_LIMIT, length)),
        },
        call_timeout=HF_CALL_TIMEOUT,
    )


async def hf_search(config: str, split: str, query: str, offset: int, length: int) -> dict:
    return await hf_get(
        "search",
        {
            "config": config,
            "split": split,
            "query": query,
            "offset": offset,
            "length": max(1, min(HF_MAX_LIMIT, length)),
        },
        call_timeout=HF_CALL_TIMEOUT,
    )


SEARCH_LOCKS: dict[str, asyncio.Lock] = {}


def match_record(record: dict, query: str) -> bool:
    needle = query.casefold()
    for field, value in record.items():
        if field == "_index":
            continue
        if needle in to_text(value).casefold():
            return True
    return False


def filter_locally(records: list, query: str) -> list:
    return [record for record in records if match_record(record, query)]


def field_hits(records: list, query: str) -> dict:
    """Is page ke records me kaunse field me match mila (kitne records me)."""
    needle = query.casefold()
    counts: dict[str, int] = {}
    for record in records:
        for field, value in record.items():
            if field == "_index":
                continue
            if needle in to_text(value).casefold():
                counts[field] = counts.get(field, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


async def scan_matches(config: str, split: str, query: str, deadline: float) -> dict:
    """
    Dataset ko chunks me padh ke locally filter karta hai — lekin DEADLINE ke andar.
    Jo mil gaya wahi return; aborted=True matlab time khatam ho gaya.
    Result cache hota hai (SEARCH_CACHE_TTL) taaki dobara wahi query fast chale.
    """
    cache_key = f"searchcache::{config}::{split}::{query.casefold()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {"records": cached, "scanned": len(cached), "complete": True, "source": "cache"}

    total_rows = int(cache_get(f"total::{config}::{split}") or 0)
    cap = min(total_rows or SCAN_MAX_ROWS, SCAN_MAX_ROWS)

    results: list = []
    scanned = 0
    complete = True
    semaphore = asyncio.Semaphore(6)
    offsets = list(range(0, cap, HF_MAX_LIMIT))

    async def grab(offset: int, call_timeout: float) -> list:
        async with semaphore:
            try:
                data = await fetch_rows(config, split, offset, HF_MAX_LIMIT,
                                        call_timeout=call_timeout)
                return normalise_records(data.get("rows", []), offset)
            except HTTPException:
                return []
            except Exception:
                return []

    for i in range(0, len(offsets), 10):
        left = deadline - time.monotonic()
        if left <= 0.4:
            complete = False
            break
        batch = offsets[i:i + 10]
        call_timeout = max(1.0, min(HF_CALL_TIMEOUT, left))
        rowsets = await asyncio.gather(*[grab(o, call_timeout) for o in batch])
        for rows in rowsets:
            scanned += len(rows)
            if len(rows) < HF_MAX_LIMIT:
                complete = True  # end of dataset
            for record in rows:
                if match_record(record, query):
                    results.append(record)
        if any(len(rows) == 0 for rows in rowsets):
            complete = False

    results.sort(key=lambda r: r.get("_index", 0))
    if complete and results:
        cache_set(cache_key, results)
    return {"records": results, "scanned": scanned, "complete": complete, "source": "scan"}


async def run_search(config: str, split: str, query: str, offset: int, length: int) -> dict:
    """
    HARD deadline wrapper: chahe HF kitna slow ho, response REQUEST_BUDGET ke andar
    hi aayega (Vercel ka 10s kill hone se pehle) — warna JSON 504 milega.
    """
    try:
        return await asyncio.wait_for(
            _run_search_inner(config, split, query, offset, length),
            timeout=REQUEST_BUDGET,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Search timeout ({REQUEST_BUDGET:.0f}s budget). Dataset bada hai — "
                "query chhota/specific karo, ya SCAN_MAX_ROWS kam karo, "
                "ya HF_TOKEN ke saath private/faster dataset use karo."
            ),
        )


async def _run_search_inner(config: str, split: str, query: str, offset: int, length: int) -> dict:
    """
    engine order: hf-filter -> local-scan (deadline ke andar) -> hf-search
    """
    started = time.monotonic()
    deadline = started + REQUEST_BUDGET

    features: list = []
    total_rows = 0
    notes: list = []
    try:
        meta = await fetch_rows(config, split, 0, 1, call_timeout=min(HF_CALL_TIMEOUT, 4.0))
        features = meta.get("features", []) or []
        total_rows = int(meta.get("num_rows_total") or 0)
    except HTTPException as exc:
        notes.append(f"schema: {exc.detail}")

    def done(records, total, feats, engine, *, scanned=0, complete=True):
        return {
            "records": records,
            "total": int(total or 0),
            "features": feats or features,
            "engine": engine,
            "took_ms": int((time.monotonic() - started) * 1000),
            "scanned": scanned,
            "complete": complete,
            "partial": not complete,
            "notes": notes[:3],
        }

    # ---- 1) HF /filter (fast server side) --------------------------------
    where = build_where(features, query)
    if where:
        try:
            data = await hf_filter(config, split, where, offset, length)
            records = normalise_records(data.get("rows", []))
            total = data.get("num_rows_total")
            return done(records, total if total is not None else len(records),
                        data.get("features"), "hf-filter",
                        scanned=total or len(records), complete=True)
        except HTTPException as exc:
            notes.append(f"filter: {exc.detail}")
            logger.info("filter unavailable (%s) -> next", exc.detail)

    # ---- 2) local scan (chhote dataset, deadline ke andar) ---------------
    if total_rows and total_rows <= SCAN_MAX_ROWS and time.monotonic() < deadline:
        try:
            scan = await scan_matches(config, split, query, deadline)
            if scan["records"] or scan["complete"]:
                return done(scan["records"][offset:offset + length], len(scan["records"]),
                            features, "local-scan",
                            scanned=scan["scanned"], complete=scan["complete"])
            notes.append("scan: kuch nahi mila / time kam pada")
        except HTTPException as exc:
            notes.append(f"scan: {exc.detail}")
            logger.info("local scan failed (%s) -> /search", exc.detail)

    # ---- 3) HF full-text search -----------------------------------------
    left = deadline - time.monotonic()
    if left <= 0.4:
        raise HTTPException(
            status_code=504,
            detail=(
                "Search timeout — dataset bada hai. Chhota/specific query try karo "
                "ya SCAN_MAX_ROWS / HF_TOKEN (private+fast) check karo."
            ),
        )
    try:
        data = await hf_search(config, split, query, offset, length)
        records = normalise_records(data.get("rows", []))
        total = data.get("num_rows_total")
        return done(records, total if total is not None else len(records),
                    data.get("features"), "hf-search",
                    scanned=total or len(records), complete=True)
    except HTTPException as exc:
        notes.append(f"search: {exc.detail}")
        raise HTTPException(
            status_code=504,
            detail=(
                "Search backend busy/timeout. 2-3 second baad retry karo. "
                f"Details: {'; '.join(notes)[:200]}"
            ),
        )


# ------------------------------------------------------- device / location --

UA_MODEL_PATTERNS = [
    r"(SM-[A-Z0-9]+)", r"(SM[A-Z0-9]{3,})", r"\b(CPH\d{4,})\b",
    r"\b(Redmi[ A-Za-z0-9.]*)", r"\b(POCO[ A-Za-z0-9.]*)", r"\b(RMX\d{4,})\b",
    r"\b(V[0-9]{4}[A-Z0-9]*)\b", r"\b(M[0-9]{4}[A-Z0-9]*)\b",
    r"\b(Pixel [0-9a-zA-Z ]+)", r"\b(OnePlus[ A-Za-z0-9]*)",
    r"\b(Moto [A-Za-z0-9]+)", r"\b(iPhone[0-9,]+)", r"\b(iPad[0-9,]+)",
]


def parse_model(ua: str) -> str:
    for pattern in UA_MODEL_PATTERNS:
        match = re.search(pattern, ua)
        if match:
            return match.group(1).strip()
    return "unknown"


def parse_os(ua: str) -> str:
    match = re.search(r"Android[ /]([0-9.]+)", ua)
    if match:
        return f"Android {match.group(1)}"
    if "Android" in ua:
        return "Android"
    match = re.search(r"(iPhone|iPad).*?OS ([0-9_]+)", ua)
    if match:
        return f"iOS {match.group(2).replace('_', '.')}"
    match = re.search(r"Windows NT ([0-9.]+)", ua)
    if match:
        return f"Windows NT {match.group(1)}"
    if "Mac OS X" in ua:
        return "macOS"
    if "Linux" in ua:
        return "Linux"
    return "unknown"


def parse_browser(ua: str) -> str:
    for token in ("Edg", "OPR", "Chrome", "Firefox", "Version", "Safari"):
        match = re.search(re.escape(token) + r"/([0-9.]+)", ua)
        if match:
            name = {"OPR": "Opera", "Edg": "Edge", "Version": "Safari"}.get(token, token)
            return f"{name}/{match.group(1)}"
    return "unknown"


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    for header in ("x-real-ip", "cf-connecting-ip", "x-vercel-forwarded-for"):
        value = request.headers.get(header)
        if value:
            return value.strip()
    return request.client.host if request.client else "unknown"


async def lookup_location(ip: str) -> dict:
    if ip in _GEO_CACHE:
        stamp, value = _GEO_CACHE[ip]
        if time.time() - stamp < 3600:
            return value

    result: dict = {}
    if LOCATION_LOOKUP and ip and not ip.startswith(("127.", "10.", "192.168.", "::1")):
        try:
            response = await asyncio.wait_for(
                app.state.client.get(
                    f"http://ip-api.com/json/{ip}",
                    params={"fields": "status,country,countryCode,regionName,city,isp,org,mobile"},
                    timeout=4.0,
                ),
                timeout=5.0,
            )
            data = response.json()
            if data.get("status") == "success":
                org = data.get("isp") or data.get("org") or ""
                if data.get("mobile"):
                    org = f"{org} (mobile)" if org else "mobile"
                result = {
                    "city": data.get("city") or "",
                    "region": data.get("regionName") or "",
                    "country": data.get("country") or "",
                    "org": org,
                }
        except Exception:
            logger.info("geo lookup failed for %s", ip)

    _GEO_CACHE[ip] = (time.time(), result)
    return result


def vercel_location(request: Request) -> dict:
    return {
        "city": request.headers.get("x-vercel-ip-city") or "",
        "region": request.headers.get("x-vercel-ip-country-region") or "",
        "country": request.headers.get("x-vercel-ip-country") or "",
        "org": "",
    }


# --------------------------------------------------------------- bot notify --

def build_bot_message(*, key, query, matches, returned, limit, via, request, ip, geo) -> str:
    ua = request.headers.get("user-agent", "")
    ua_short = ua if len(ua) <= 96 else ua[:96] + "…"
    location = ", ".join([p for p in (geo.get("city"), geo.get("region"), geo.get("country")) if p])
    if geo.get("org"):
        location = f"{location} · {geo['org']}" if location else geo["org"]

    device_fp = hashlib.sha256(
        f"{ua}|{ip}|{request.headers.get('accept-language', '')}".encode("utf-8")
    ).hexdigest()[:16]

    query_line = f'q -> any field (partial) = "{query}"' if query else "q -> none (browse mode)"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return "\n".join([
        f"🔎 SEARCH — {API_TITLE}",
        "",
        f"👤 Key: {mask_key(key)} (id {key_id(key)})",
        "",
        f"🧭 Query: {query_line}",
        "",
        f"📊 Matches: {matches} (returned {returned}, limit {limit})",
        "",
        f"🛠 Via: {via}",
        "",
        f"📍 Location: {location or 'unknown'}",
        f"🌐 IP: {ip}",
        f"🖨 Device FP: {device_fp}",
        f"🖥 UA: {ua_short or 'unknown'}",
        f"📱 Model: {parse_model(ua)}",
        f"⚙️ OS: {parse_os(ua)} · {'mobile' if 'Mobile' in ua else 'desktop'}",
        f"🌐 Browser: {parse_browser(ua)}",
        f"🕒 {now}",
    ])


async def send_bot(text: str) -> None:
    if not BOT_ENABLED:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        for i in range(0, len(text), 3500):
            await asyncio.wait_for(
                app.state.client.post(
                    url,
                    json={
                        "chat_id": BOT_CHAT_ID,
                        "text": text[i:i + 3500],
                        "disable_web_page_preview": True,
                    },
                    timeout=5.0,
                ),
                timeout=6.0,
            )
    except Exception:
        logger.exception("bot notification failed")


async def notify(**kwargs) -> None:
    try:
        await send_bot(build_bot_message(**kwargs))
    except Exception:
        logger.exception("notification build failed")


# ------------------------------------------------------------ shared worker --

def wants_html(request: Request, fmt: Optional[str], ui: Optional[int]) -> bool:
    if fmt == "json" or fmt == "compact":
        return False
    if ui == 1 or fmt == "html":
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept


def resolve_paging(page: int, limit: int, offset: Optional[int]) -> tuple[int, int]:
    limit = max(1, min(MAX_PAGE_SIZE, int(limit or PAGE_SIZE)))
    if offset is None:
        offset = (max(1, int(page or 1)) - 1) * limit
    return max(0, int(offset)), limit


async def fetch_data(
    request: Request,
    key: str,
    query: str,
    offset: int,
    limit: int,
    config: Optional[str],
    split: Optional[str],
    notify_flag: bool,
    via: str,
) -> dict:
    """Data layer: key check -> split resolve -> rows/search -> notify."""
    checked = check_key(key)
    key = checked["key"]

    use_config = (config or HF_CONFIG).strip() or HF_CONFIG
    use_split = (split or HF_SPLIT).strip() or HF_SPLIT
    resolved = await resolve_split(use_config, use_split)
    use_config = resolved.get("config", use_config)
    use_split = resolved.get("split", use_split)

    query = (query or "").strip()
    features = cache_get(f"schema::{use_config}::{use_split}") or []
    extra: dict = {}

    if query:
        result = await run_search(use_config, use_split, query, offset, limit)
        records, total = result["records"], result["total"]
        features = result["features"] or features
        engine, took_ms = result["engine"], result["took_ms"]
        extra = {
            "scanned": result.get("scanned", 0),
            "complete": result.get("complete", True),
            "partial": result.get("partial", False),
            "notes": result.get("notes", []),
        }
    else:
        data = await fetch_rows(use_config, use_split, offset, limit)
        records = normalise_records(data.get("rows", []), offset)
        total = int(data.get("num_rows_total") or 0)
        features = data.get("features", []) or features
        engine, took_ms = "hf-rows", 0
        cache_set(f"schema::{use_config}::{use_split}", features)
        cache_set(f"total::{use_config}::{use_split}", total)

    if notify_flag:
        ip = client_ip(request)
        geo = vercel_location(request)
        if geo.get("city") or geo.get("country"):
            geo_extra = await lookup_location(ip)
            if geo_extra.get("org"):
                geo["org"] = geo_extra["org"]
        else:
            geo = await lookup_location(ip)
        await notify(
            key=key, query=query, matches=total, returned=len(records),
            limit=limit, via=via, request=request, ip=ip, geo=geo,
        )

    return {
        "key": key,
        "query": query,
        "records": records,
        **extra,
        "total": total,
        "offset": offset,
        "limit": limit,
        "config": use_config,
        "split": use_split,
        "schema": features_to_schema(features),
        "engine": engine,
        "took_ms": took_ms,
    }


def paging_block(data: dict) -> dict:
    offset, limit, total = data["offset"], data["limit"], data["total"]
    count = len(data["records"])
    page = (offset // limit) + 1 if limit else 1
    has_more = (offset + count) < total

    def url(page_no: int) -> str:
        base = f"/{data['key']}" + (f"/{data['query']}" if data["query"] else "")
        return f"{base}?page={page_no}&limit={limit}"

    return {
        "page": page,
        "limit": limit,
        "offset": offset,
        "count": count,
        "total": total,
        "total_pages": (total + limit - 1) // limit if limit and total else 0,
        "has_more": has_more,
        "next_page_url": url(page + 1) if has_more else None,
        "prev_page_url": url(page - 1) if page > 1 else None,
    }


# ============================== ROUTE 1: home ===============================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HOME_HTML)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset_configured": bool(HF_DATASET),
        "dataset": HF_DATASET or None,
        "bot_enabled": BOT_ENABLED,
        "keys": {
            "restricted": not ALLOW_ALL_KEYS,
            "configured": len(ALLOWED_KEYS),
            "masked": [mask_key(k) for k in ALLOWED_KEYS][:10],
        },
        "limits": {"page_size": PAGE_SIZE, "max_page_size": MAX_PAGE_SIZE, "hf_max": HF_MAX_LIMIT},
        "time": datetime.now(timezone.utc).isoformat(),
    }


# ============================== ROUTE 2: metadata ==========================

@app.get("/api/dataset")
async def api_dataset():
    splits = await get_splits()
    resolved = await resolve_split(HF_CONFIG, HF_SPLIT)
    features = await get_schema(resolved["config"], resolved["split"])
    return {
        "success": True,
        "endpoint": "dataset-info",
        "dataset": HF_DATASET,
        "splits": splits,
        "active": {"config": resolved["config"], "split": resolved["split"]},
        "total_rows": len(splits) and cache_get(f"total::{resolved['config']}::{resolved['split']}"),
        "column_count": len(features),
    }


@app.get("/api/fields")
async def api_fields(config: Optional[str] = None, split: Optional[str] = None):
    resolved = await resolve_split(config or HF_CONFIG, split or HF_SPLIT)
    features = await get_schema(resolved["config"], resolved["split"])
    total = cache_get(f"total::{resolved['config']}::{resolved['split']}") or 0
    return {
        "success": True,
        "endpoint": "fields",
        "config": resolved["config"],
        "split": resolved["split"],
        "total_rows": total,
        "column_count": len(features),
        "columns": features_to_schema(features),
    }


@app.get("/api/key/{key}")
async def api_key_check(key: str):
    """Sirf key validation — koi dataset data nahi."""
    checked = check_key(key)
    real = checked["key"]
    resolved = await resolve_split(HF_CONFIG, HF_SPLIT)
    features = await get_schema(resolved["config"], resolved["split"])
    return {
        "success": True,
        "endpoint": "key-check",
        "key": real,
        "masked": mask_key(real),
        "id": key_id(real),
        "allowed": True,
        "restricted_mode": checked["restricted"],
        "dataset": HF_DATASET,
        "total_rows": cache_get(f"total::{resolved['config']}::{resolved['split']}") or 0,
        "column_count": len(features),
        "routes": {
            "browse": f"/{real}",
            "search": f"/{real}/{{query}}",
            "ui": f"/{real}?ui=1",
            "json": f"/{real}?format=json",
        },
    }


# ============================== ROUTE 3: path browse =======================

@app.get("/{key}")
async def path_browse(
    request: Request,
    key: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=HF_MAX_LIMIT),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = Query(default=None, pattern="^(json|compact|html)$"),
    ui: Optional[int] = Query(default=None, ge=0, le=1),
    notify: int = Query(default=1, ge=0, le=1),
):
    if wants_html(request, format, ui):
        return HTMLResponse(DSL_HTML)

    off, lim = resolve_paging(page, limit, offset)
    data = await fetch_data(request, key, "", off, lim, config, split,
                            bool(notify), "path browse (/key)")
    return pretty({
        "success": True,
        "endpoint": "path-browse",
        "mode": "browse",
        "key": data["key"],
        "dataset": HF_DATASET,
        "config": data["config"],
        "split": data["split"],
        "engine": data["engine"],
        "pagination": paging_block(data),
        "schema": data["schema"],
        "results": data["records"],
    }, compact=(format == "compact"))


# ============================== ROUTE 4: path search =======================

@app.get("/{key}/{query}")
async def path_search(
    request: Request,
    key: str,
    query: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=HF_MAX_LIMIT),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = Query(default=None, pattern="^(json|compact|html)$"),
    ui: Optional[int] = Query(default=None, ge=0, le=1),
    notify: int = Query(default=1, ge=0, le=1),
):
    if wants_html(request, format, ui):
        return HTMLResponse(DSL_HTML)

    off, lim = resolve_paging(page, limit, offset)
    data = await fetch_data(request, key, query, off, lim, config, split,
                            bool(notify), "path search (/key/query)")
    return pretty({
        "success": True,
        "endpoint": "path-search",
        "mode": "search",
        "key": data["key"],
        "dataset": HF_DATASET,
        "config": data["config"],
        "split": data["split"],
        "search": {
            "query": data["query"],
            "matched": data["total"] > 0,
            "engine": data["engine"],
            "fields_searched": [c["name"] for c in data["schema"]],
            "took_ms": data["took_ms"],
            "scanned": data.get("scanned", 0),
            "complete": data.get("complete", True),
            "partial": data.get("partial", False),
            "notes": data.get("notes", []),
            "field_hits_on_page": field_hits(data["records"], data["query"]),
        },
        "pagination": paging_block(data),
        "schema": data["schema"],
        "results": data["records"],
    }, compact=(format == "compact"))


# ============================== ROUTE 5: API v1 browse =====================

@app.get("/api/v1/{key}")
async def api_browse(
    request: Request,
    key: str,
    cursor: Optional[int] = Query(default=None, ge=0, description="offset based cursor"),
    limit: int = Query(default=MAX_PAGE_SIZE, ge=1, le=HF_MAX_LIMIT),
    config: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = Query(default=None, pattern="^(json|compact|html)$"),
    ui: Optional[int] = Query(default=None, ge=0, le=1),
    notify: int = Query(default=1, ge=0, le=1),
):
    if wants_html(request, format, ui):
        return HTMLResponse(DSL_HTML)

    off, lim = (cursor or 0), max(1, min(MAX_PAGE_SIZE, limit))
    data = await fetch_data(request, key, "", off, lim, config, split,
                            bool(notify), "api browse (/api/v1/key)")
    next_cursor = off + len(data["records"])
    has_more = next_cursor < data["total"]

    return pretty({
        "ok": True,
        "endpoint": "api-browse",
        "key": data["key"],
        "dataset": HF_DATASET,
        "cursor": {
            "offset": off,
            "limit": lim,
            "next_cursor": next_cursor if has_more else None,
            "has_more": has_more,
            "total": data["total"],
        },
        "columns": [c["name"] for c in data["schema"]],
        "items": data["records"],
    }, compact=(format == "compact"))


# ============================== ROUTE 6: API v1 search =====================

@app.get("/api/v1/{key}/{query}")
async def api_search(
    request: Request,
    key: str,
    query: str,
    cursor: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(default=MAX_PAGE_SIZE, ge=1, le=HF_MAX_LIMIT),
    config: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = Query(default=None, pattern="^(json|compact|html)$"),
    ui: Optional[int] = Query(default=None, ge=0, le=1),
    notify: int = Query(default=1, ge=0, le=1),
):
    if wants_html(request, format, ui):
        return HTMLResponse(DSL_HTML)

    off, lim = (cursor or 0), max(1, min(MAX_PAGE_SIZE, limit))
    data = await fetch_data(request, key, query, off, lim, config, split,
                            bool(notify), "api search (/api/v1/key/query)")
    next_cursor = off + len(data["records"])
    has_more = next_cursor < data["total"]

    return pretty({
        "ok": True,
        "endpoint": "api-search",
        "key": data["key"],
        "query": data["query"],
        "total_matches": data["total"],
        "cursor": {
            "offset": off,
            "limit": lim,
            "next_cursor": next_cursor if has_more else None,
            "has_more": has_more,
        },
        "engine": data["engine"],
        "took_ms": data["took_ms"],
        "scanned": data.get("scanned", 0),
        "complete": data.get("complete", True),
        "partial": data.get("partial", False),
        "field_hits": field_hits(data["records"], data["query"]),
        "items": data["records"],
    }, compact=(format == "compact"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "endpoint": "error",
            "status": exc.status_code,
            "error": exc.detail,
        },
    )


# ================================== HTML ===================================

DOCS_HTML = """
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset JSON API</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;padding:28px 16px 70px;background:#0b0f1a;color:#e6e9f2;
   font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
 .wrap{max-width:900px;margin:0 auto}
 h1{font-size:22px;margin:0 0 6px}
 .sub{color:#8b93a7;font-size:13px;margin-bottom:22px}
 table{width:100%;border-collapse:collapse;font-size:13.5px}
 th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #1b2237;vertical-align:top}
 th{color:#8b93a7;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
 code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#9ecbff;font-size:12.5px}
 .card{background:#111827;border:1px solid #1f2740;border-radius:14px;padding:18px;margin:16px 0}
 .muted{color:#8b93a7;font-size:13px}
 .pill{display:inline-block;font-size:11px;padding:2px 9px;border-radius:999px;background:#1b2740;color:#9fb4dd}
</style></head><body><div class="wrap">
<h1>Dataset JSON API <span class="pill">per-endpoint alag response</span></h1>
<div class="sub">Har endpoint ka kaam aur response shape alag hai. URL me <code>{key}</code> aapki API key hai.</div>
<div class="card">
<table>
<tr><th>Route</th><th>Kaam</th></tr>
<tr><td><code>/</code></td><td>Ye page (docs)</td></tr>
<tr><td><code>/health</code></td><td>status + keys + limits</td></tr>
<tr><td><code>/api/dataset</code></td><td>configs / splits / total rows</td></tr>
<tr><td><code>/api/fields</code></td><td>schema: column + dtype</td></tr>
<tr><td><code>/api/key/{key}</code></td><td>sirf key check (data nahi)</td></tr>
<tr><td><code>/{key}</code></td><td><b>browse</b> data — page based : <code>pagination</code> + <code>results</code></td></tr>
<tr><td><code>/{key}/{query}</code></td><td><b>search</b> data — <code>search</code> meta + <code>pagination</code> + <code>results</code></td></tr>
<tr><td><code>/api/v1/{key}</code></td><td>browse — <b>cursor</b> based : <code>cursor</code> + <code>items</code></td></tr>
<tr><td><code>/api/v1/{key}/{query}</code></td><td>search — <code>field_hits</code> + <code>cursor</code> + <code>items</code></td></tr>
</table>
</div>
<div class="card">
<div class="muted">Params (data routes par):</div>
<p><code>?ui=1</code> infinite-scroll UI · <code>?format=json</code> pretty JSON ·
<code>?format=compact</code> compact · <code>?limit=100&amp;page=2</code> paging ·
<code>?notify=0</code> bot off</p>
<div class="muted">Example:</div>
<p><code>/14ec092a?ui=1</code><br><code>/14ec092a/rahul?format=json&amp;limit=50</code><br>
<code>/api/v1/14ec092a?cursor=0&amp;limit=100</code></p>
</div>
</div></body></html>
"""

DSL_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JSON Data — infinite scroll</title>
<style>
 :root{color-scheme:dark;--bg:#0a0e18;--card:#111827;--line:#1f2740;--txt:#e6e9f2;--mut:#8b93a7;--acc:#2b6ef6}
 *{box-sizing:border-box}
 body{margin:0;padding:0 0 90px;background:var(--bg);color:var(--txt);
   font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
 header{position:sticky;top:0;z-index:20;backdrop-filter:blur(10px);
   background:rgba(10,14,24,.88);border-bottom:1px solid var(--line);padding:12px 16px}
 .wrap{max-width:1080px;margin:0 auto}
 .top{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between}
 h1{font-size:15px;margin:0;font-weight:700;letter-spacing:.2px}
 h1 span{color:var(--mut);font-weight:500}
 .pills{display:flex;gap:6px;flex-wrap:wrap}
 .pill{font-size:11px;padding:3px 9px;border-radius:999px;background:#16203a;color:#9fb4dd;
   border:1px solid var(--line);font-family:ui-monospace,Menlo,monospace}
 .pill b{color:#e6e9f2;font-weight:600}
 .bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
 input,select,button,a.btn{font:inherit;border-radius:10px;border:1px solid var(--line);
   background:#0d1424;color:var(--txt);padding:8px 11px;outline:none}
 input:focus,select:focus{border-color:var(--acc)}
 button,a.btn{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600;
   cursor:pointer;text-decoration:none;display:inline-block}
 button.ghost,a.btn.ghost{background:#0d1424;color:var(--txt);border-color:var(--line);font-weight:500}
 #q{flex:1 1 240px;min-width:180px}
 main{max-width:1080px;margin:0 auto;padding:16px}
 .rec{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px;overflow:hidden}
 .rec .head{display:flex;justify-content:space-between;gap:10px;padding:8px 12px;
   border-bottom:1px solid var(--line);font-size:11.5px;color:var(--mut);
   font-family:ui-monospace,Menlo,monospace}
 .rec pre{margin:0;padding:12px;overflow:auto;font-size:12.5px;line-height:1.5;
   font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;tab-size:2}
 .k{color:#7fb0ff}.s{color:#a5e0a5}.n{color:#f0b26b}.b{color:#d8a0f0}
 .state{text-align:center;padding:24px 12px;color:var(--mut);font-size:13px}
 .err{background:#2a1420;border:1px solid #5a2233;color:#ffb3c1;border-radius:12px;
   padding:14px;margin:12px 0;font-size:13px;white-space:pre-wrap;word-break:break-word}
 .empty{text-align:center;color:var(--mut);padding:40px 10px}
 @keyframes sp{to{transform:rotate(360deg)}}
 #top{position:fixed;right:16px;bottom:16px;border-radius:50%;width:44px;height:44px;
   padding:0;display:none;font-size:16px}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="top">
    <h1>JSON Data <span id="ttl">—</span></h1>
    <div class="pills" id="pills"></div>
  </div>
  <div class="bar">
    <input id="q" placeholder="Search any field… (Enter)">
    <select id="limit"><option value="20">20 / page</option><option value="50">50 / page</option>
      <option value="100" selected>100 / page</option></select>
    <button id="go">Search</button>
    <button id="retry" class="ghost" style="display:none">Retry</button>
    <span id="spin" style="display:none;width:14px;height:14px;border-radius:50%;
      border:2px solid #2b6ef6;border-top-color:transparent;animation:sp .7s linear infinite"></span>
    <a class="btn ghost" id="rawlink" href="#" target="_blank">Raw JSON</a>
    <a class="btn ghost" id="prevlink" href="/">Docs</a>
  </div>
</div></header>

<main id="out"></main>
<div class="state" id="loading"></div>
<div id="sentinel"></div>
<button id="top">↑</button>

<script>
const $ = id => document.getElementById(id);
const S = {
  key:"", query:"", limit:100, offset:0, total:0, loaded:0,
  engine:"", done:true, busy:false, pages:0,
  fails:0, resultSig:null, errThrottle:0, api:false
};

function parseRoute(){
  const parts = location.pathname.split("/").filter(Boolean);
  let api = false;
  if (parts[0] === "api" && parts[1] === "v1"){ api = true; parts.splice(0,2); }
  return { api,
    key: parts[0] ? decodeURIComponent(parts[0]) : "",
    query: parts[1] ? decodeURIComponent(parts[1]) : "" };
}

const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function hl(json){
  return esc(json).replace(/("[^"\n]*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?/g,
    (m,str,colon,lit)=>{
      if (str) return `<span class="${colon?"k":"s"}">${str}</span>${colon||""}`;
      if (lit) return `<span class="b">${m}</span>`;
      return `<span class="n">${m}</span>`;
    });
}

function recordCard(rec){
  const copy = {...rec};
  const idx = copy._index; delete copy._index;
  return `<div class="rec"><div class="head"><span>record${idx!==undefined?" #"+idx:""}</span>
    <span>${Object.keys(copy).length} fields</span></div>
    <pre>${hl(JSON.stringify(copy,null,2))}</pre></div>`;
}

function busy(on){ S.busy = on; $("spin").style.display = on ? "inline-block" : "none"; }

function stopAll(msg){
  S.done = true;
  $("loading").textContent = msg || "";
  busy(false);
}

function errBox(msg){
  // ek hi error card, max 2 second me ek baar (loop se bachne ke liye)
  const now = Date.now();
  if (now - S.errThrottle < 2000) return;
  S.errThrottle = now;
  const old = document.querySelectorAll(".err");
  old.forEach((el, i) => { if (i < old.length - 1) el.remove(); });
  $("out").insertAdjacentHTML("afterbegin", `<div class="err">${esc(msg)}</div>`);
}

async function fetchPage(){
  const page = S.pages + 1;
  const base = S.api ? "/api/v1/" : "/";
  const url = base + encodeURIComponent(S.key) + (S.query ? "/" + encodeURIComponent(S.query) : "")
    + (S.api ? `?format=json&cursor=${S.offset}&limit=${S.limit}`
             : `?format=json&page=${page}&limit=${S.limit}`);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  let res, text;
  try {
    res = await fetch(url, { headers: { "Accept": "application/json" }, signal: controller.signal });
    text = await res.text();
  } catch (e) {
    clearTimeout(timer);
    throw new Error(controller.signal.aborted
      ? "Request timeout (20s). Search chhota karo ya retry button dabao."
      : "Network error: " + e.message);
  }
  clearTimeout(timer);

  let data;
  try { data = JSON.parse(text); }
  catch (e) { throw new Error("Server ne JSON nahi diya (HTTP " + res.status + "): " + text.slice(0, 160)); }

  if (!res.ok || data.success === false || data.ok === false) {
    throw new Error((data.error || ("HTTP " + res.status)) + "\n\n(Retry karne ke liye niche ‘Retry’ dabao — page automatically loop nahi karega.)");
  }
  return data;
}

function normalize(data){
  const pg = data.pagination || {}, cur = data.cursor || {};
  const items = data.items || data.results || [];
  return {
    items,
    total: pg.total ?? cur.total ?? data.total_matches ?? data.total ?? 0,
    has_more: pg.has_more ?? cur.has_more ?? false,
    engine: data.engine || (data.search && data.search.engine) || "-",
    endpoint: data.endpoint || "",
    partial: data.partial ?? (data.search && data.search.partial) ?? false
  };
}

function paint(page){
  $("ttl").textContent = S.query ? "search: " + S.query : "browse";
  $("pills").innerHTML =
    `<span class="pill">key <b>${esc(S.key)}</b></span>` +
    `<span class="pill">endpoint <b>${esc(S.endpoint)}</b></span>` +
    `<span class="pill">engine <b>${esc(S.engine)}</b></span>` +
    `<span class="pill">loaded <b>${S.loaded}</b> / ${S.total}</span>` +
    (S.query ? `<span class="pill">query <b>${esc(S.query)}</b></span>` : "") +
    (page && page.partial ? `<span class="pill">partial scan</span>` : "");
  const base = S.api ? `/api/v1/${encodeURIComponent(S.key)}` : `/${encodeURIComponent(S.key)}`;
  $("rawlink").href = base + (S.query ? `/${encodeURIComponent(S.query)}` : "") + "?format=json&limit=" + S.limit;
}

async function loadMore(){
  if (S.busy || S.done) return;
  busy(true);
  $("loading").textContent = "Loading " + (S.query ? "results" : "page " + (S.pages + 1)) + "…";
  try {
    const page = normalize(await fetchPage());

    // ---- guard: same page dobara nahi (loop se bachne ke liye) ----
    const sig = (page.items[0] && page.items[0]._index) + ":" + page.items.length + ":" + page.total;
    if (S.resultSig !== null && sig === S.resultSig) { stopAll("— end of data —"); return; }
    S.resultSig = sig;

    S.total = page.total; S.engine = page.engine; S.endpoint = page.endpoint;
    S.offset += S.limit; S.pages += 1; S.loaded += page.items.length;
    S.fails = 0;
    S.done = !page.has_more || page.items.length === 0;

    if (S.pages === 1 && page.items.length === 0) {
      $("out").innerHTML = `<div class="empty">Kuch nahi mila${S.query?` for “${esc(S.query)}”`:""}.</div>`;
    }
    $("out").insertAdjacentHTML("beforeend", page.items.map(recordCard).join(""));
    paint(page);
    $("loading").textContent = S.done ? (S.loaded ? "— end of data —" : "") : "";
    $("retry").style.display = "none";
  } catch (err) {
    S.fails += 1;
    errBox(err.message);
    if (S.fails >= 2) {
      stopAll("Ruk gaya (error). ‘Retry’ dabao.");
      $("retry").style.display = "inline-block";
    } else {
      // sirf ek baar, thodi der baad
      $("loading").textContent = "Error — ek baar phir koshish…";
      setTimeout(() => { if (!S.done) loadMore(); }, 1500);
    }
  } finally {
    busy(false);
  }
}

function reset(key, query){
  S.key = key; S.query = query; S.limit = parseInt($("limit").value,10);
  S.offset = 0; S.pages = 0; S.loaded = 0; S.total = 0;
  S.done = false; S.busy = false; S.fails = 0; S.resultSig = null; S.errThrottle = 0;
  $("out").innerHTML = ""; $("loading").textContent = "";
  $("retry").style.display = "none";
  loadMore();
}

$("go").onclick = () => {
  const q = $("q").value.trim();
  const base = (S.api ? "/api/v1/" : "/") + encodeURIComponent(S.key);
  history.pushState({}, "", q ? base + "/" + encodeURIComponent(q) + "?ui=1" : base + "?ui=1");
  reset(S.key, q);
};
$("q").addEventListener("keydown", e => { if (e.key === "Enter") $("go").click(); });
$("limit").onchange = () => reset(S.key, S.query);
$("retry").onclick = () => {
  S.fails = 0; S.resultSig = null; S.done = false;
  document.querySelectorAll(".err").forEach(el => el.remove());
  loadMore();
};
$("top").onclick = () => window.scrollTo({ top:0, behavior:"smooth" });
window.addEventListener("scroll", () => { $("top").style.display = scrollY > 600 ? "block" : "none"; });

// ---- infinite scroll: sirf tab jab busy nahi + done nahi + error nahi ----
let ioLock = false;
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting || ioLock) return;
  if (S.busy || S.done || S.fails >= 2) return;
  ioLock = true;
  loadMore().finally(() => setTimeout(() => { ioLock = false; }, 400));
}, { rootMargin: "500px 0px" }).observe($("sentinel"));

const route = parseRoute();
if (!route.key) { location.replace("/"); }
else {
  S.api = route.api;
  $("q").value = route.query;
  reset(route.key, route.query);
}
</script>
</body>
</html>
"""

HOME_HTML = DOCS_HTML


# Vercel Python runtime entry point: `app` auto-detect hota hai.
