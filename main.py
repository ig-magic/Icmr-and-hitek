"""
================================================================================
 HF DATASET API  —  single file (Vercel Python runtime ready)
================================================================================
Routes
------
  GET  /                       -> chhota docs/explorer page
  GET  /health                 -> health check
  GET  /api/dataset            -> configs / splits / features info
  GET  /api/v1/{key}           -> JSON, paginated (infinite scroll friendly)
  GET  /api/v1/{key}/{query}   -> JSON, search across ALL fields (partial match)
  GET  /{key}                  -> same as /api/v1/{key}
  GET  /{key}/{query}          -> same as /api/v1/{key}/{query}

Pagination query params (dono endpoints par):
  page=1        (1-based)
  limit=20      (1..100)
  offset=0      (page ka alternative)
  config=default split=train   (optional override)
  notify=0      (bot notification band karna ho to)

Env vars (Vercel -> Settings -> Environment Variables)
------------------------------------------------------
  HF_DATASET        = username/dataset-name          (REQUIRED)
  HF_TOKEN          = hf_xxx                         (private dataset ke liye)
  HF_CONFIG         = default
  HF_SPLIT          = train
  ALLOWED_KEYS      = key1,key2,key3                 (optional; khaali = sab keys allowed)
  PAGE_SIZE         = 20
  MAX_PAGE_SIZE     = 100
  CACHE_TTL         = 120      (seconds, dataset rows cache)
  SCAN_MAX_ROWS     = 20000    (search fallback me max rows)
  BOT_TOKEN         = 123456:ABC...                  (Telegram bot token)
  BOT_CHAT_ID       = 123456789                      (owner chat id)
  BOT_SILENT        = 0        (1 = notification fail ho to chup raho)
  API_TITLE         = my Student Data API            (bot message heading)
  LOCATION_LOOKUP   = 1        (IP se location/org nikalna)
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
from fastapi.responses import HTMLResponse, JSONResponse

# ------------------------------------------------------------------ config ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-dataset-api")


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
TIMEOUT = httpx.Timeout(25.0, connect=8.0)

PAGE_SIZE = _env_int("PAGE_SIZE", 20, 1, 100)
MAX_PAGE_SIZE = _env_int("MAX_PAGE_SIZE", 100, 1, 100)
CACHE_TTL = _env_int("CACHE_TTL", 120, 0, 86400)
SCAN_MAX_ROWS = _env_int("SCAN_MAX_ROWS", 20000, 100, 500000)

ALLOWED_KEYS = [k.strip() for k in _env("ALLOWED_KEYS").split(",") if k.strip()]

BOT_TOKEN = _env("BOT_TOKEN") or _env("TELEGRAM_BOT_TOKEN")
BOT_CHAT_ID = _env("BOT_CHAT_ID") or _env("TELEGRAM_CHAT_ID")
BOT_SILENT = _env("BOT_SILENT", "0") == "1"
API_TITLE = _env("API_TITLE", "my Student Data API") or "my Student Data API"
LOCATION_LOOKUP = _env("LOCATION_LOOKUP", "1") == "1"

BOT_ENABLED = bool(BOT_TOKEN and BOT_CHAT_ID)

# ----------------------------------------------------------------- helpers ---

_CACHE: dict[str, tuple[float, Any]] = {}
_GEO_CACHE: dict[str, tuple[float, dict]] = {}


def cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    stamp, value = item
    if CACHE_TTL and (time.time() - stamp) > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    if not CACHE_TTL:
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    if CACHE_TTL:
        _CACHE[key] = (time.time(), value)


def get_headers() -> dict:
    headers = {"User-Agent": "HF-Dataset-Explorer/2.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def mask_key(key: str) -> str:
    """14ec092a -> 14****2a  (chhota key -> ****)"""
    if len(key) <= 4:
        return "****"
    return f"{key[:2]}{'*' * min(6, max(2, len(key) - 4))}{key[-2:]}"


def short_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def clean(value: Any) -> Any:
    """JSON-safe banao (datetime/bytes etc)."""
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


# ------------------------------------------------------------ HF requests ---

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
    title="HF Dataset JSON API",
    version="2.0.0",
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


async def hf_get(endpoint: str, params: dict) -> dict:
    if not HF_DATASET:
        raise HTTPException(
            status_code=503,
            detail="Set the HF_DATASET environment variable.",
        )

    try:
        response = await app.state.client.get(
            f"{HF_API}/{endpoint.lstrip('/')}",
            params={"dataset": HF_DATASET, **params},
        )
    except httpx.TimeoutException:
        logger.exception("Hugging Face request timed out (%s)", endpoint)
        raise HTTPException(status_code=504, detail="Hugging Face request timed out. Try again.")
    except httpx.RequestError:
        logger.exception("Hugging Face network request failed (%s)", endpoint)
        raise HTTPException(status_code=502, detail="Could not connect to Hugging Face.")

    if response.status_code >= 400:
        body = ""
        try:
            body = response.text[:300]
        except Exception:
            pass
        logger.error("HF HTTP %s on %s :: %s", response.status_code, endpoint, body)
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API returned HTTP {response.status_code}.",
        )

    try:
        data = response.json()
    except ValueError:
        logger.exception("Invalid JSON from Hugging Face (%s)", endpoint)
        raise HTTPException(status_code=502, detail="Hugging Face returned invalid JSON.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected response from Hugging Face.")

    if data.get("error"):
        logger.warning("Hugging Face dataset error: %s", data["error"])
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
    """Config/split validate karo; invalid ho to pehla available use karo."""
    splits = await get_splits()

    for item in splits:
        if item.get("config") == config and item.get("split") == split:
            return item

    # fallback: agar dataset me config/split naam different hai
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
        detail="Invalid config or split. Check /api/dataset for available values.",
    )


async def fetch_rows(config: str, split: str, offset: int, length: int) -> dict:
    length = max(1, min(100, length))
    return await hf_get(
        "rows",
        {"config": config, "split": split, "offset": offset, "length": length},
    )


def normalise_records(rows: list, start_index: Optional[int] = None) -> list:
    """[{row:{...},row_idx:n}] -> [{...fields, _index:n}]"""
    out = []
    for i, item in enumerate(rows or []):
        row = item.get("row") if isinstance(item, dict) else None
        if not isinstance(row, dict):
            row = item if isinstance(item, dict) else {"value": item}
        record = clean(row)
        idx = item.get("row_idx", None) if isinstance(item, dict) else None
        if idx is None and start_index is not None:
            idx = start_index + i
        if idx is not None:
            record["_index"] = idx
        out.append(record)
    return out


def feature_names(features: list) -> list:
    return [f.get("name") for f in (features or []) if f.get("name")]


# ------------------------------------------------------------- searching ----

def build_where(features: list, query: str) -> Optional[str]:
    """Sabhi fields par partial (ILIKE) match ka SQL where clause."""
    q = query.replace("'", "''")
    parts = []
    for feature in features or []:
        name = feature.get("name")
        dtype = ((feature.get("type") or {}).get("dtype") or "string").lower()
        if not name:
            continue
        if any(ch in name for ch in ('"', "\\", "\n")):
            continue
        col = f'"{name}"'
        if dtype == "string":
            parts.append(f"{col} ILIKE '%{q}%'")
        elif dtype in ("int8", "int16", "int32", "int64",
                       "uint8", "uint16", "uint32", "uint64",
                       "float16", "float32", "float64", "bool", "boolean"):
            parts.append(f"CAST({col} AS VARCHAR) ILIKE '%{q}%'")
    if not parts:
        return None
    return " OR ".join(parts)


async def hf_filter(config: str, split: str, where: str, offset: int, length: int) -> dict:
    return await hf_get(
        "filter",
        {
            "config": config,
            "split": split,
            "where": where,
            "offset": offset,
            "length": max(1, min(100, length)),
        },
    )


async def hf_search(config: str, split: str, query: str, offset: int, length: int) -> dict:
    return await hf_get(
        "search",
        {
            "config": config,
            "split": split,
            "query": query,
            "offset": offset,
            "length": max(1, min(100, length)),
        },
    )


async def scan_all_rows(config: str, split: str) -> dict:
    """Fallback: poora dataset padho (cache ke saath) phir python me filter karo."""
    cache_key = f"scan::{config}::{split}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    first = await fetch_rows(config, split, 0, 100)
    features = first.get("features", []) or []
    total = first.get("num_rows_total") or 0
    limit = min(int(total or 0) or SCAN_MAX_ROWS, SCAN_MAX_ROWS)

    records = normalise_records(first.get("rows", []), 0)

    chunk = 100
    semaphore = asyncio.Semaphore(6)

    async def grab(offset: int) -> list:
        async with semaphore:
            try:
                data = await fetch_rows(config, split, offset, chunk)
                return normalise_records(data.get("rows", []), offset)
            except HTTPException:
                return []

    offsets = list(range(chunk, limit, chunk))
    for i in range(0, len(offsets), 12):
        batches = await asyncio.gather(*[grab(o) for o in offsets[i:i + 12]])
        for batch in batches:
            records.extend(batch)

    records.sort(key=lambda r: r.get("_index", 0))
    payload = {"features": features, "records": records, "total": total}
    cache_set(cache_key, payload)
    return payload


def filter_locally(records: list, query: str) -> list:
    needle = query.casefold()
    hits = []
    for record in records:
        for key, value in record.items():
            if key == "_index":
                continue
            if needle in to_text(value).casefold():
                hits.append(record)
                break
    return hits


async def run_search(config: str, split: str, query: str, offset: int, length: int) -> dict:
    """
    Returns:
      {records, total, features, engine}

    Strategy (best -> last resort):
      1) HF /filter with  CAST(col AS VARCHAR) ILIKE '%q%'  on EVERY field
         -> fastest + true partial, case-insensitive, numbers bhi match
      2) local scan (dataset chhota ho to) -> 100% exact, koi engine limit nahi
      3) HF /search (full-text) -> bade datasets ke liye last option
    """
    features: list = []
    total_rows = 0
    engine = "hf-filter"

    try:
        meta = await fetch_rows(config, split, 0, 1)
        features = meta.get("features", []) or []
        total_rows = int(meta.get("num_rows_total") or 0)
    except HTTPException:
        features = []

    # ---- 1) Hugging Face /filter -----------------------------------------
    where = build_where(features, query)
    if where:
        try:
            data = await hf_filter(config, split, where, offset, length)
            records = normalise_records(data.get("rows", []))
            total = data.get("num_rows_total")
            return {
                "records": records,
                "total": int(total if total is not None else len(records)),
                "features": data.get("features", features) or features,
                "engine": engine,
            }
        except HTTPException as exc:
            logger.info("filter unavailable (%s) -> next strategy", exc.detail)

    # ---- 2) local scan (chhote/medium datasets) --------------------------
    if total_rows and total_rows <= SCAN_MAX_ROWS:
        try:
            scan = await scan_all_rows(config, split)
            hits = filter_locally(scan["records"], query)
            return {
                "records": hits[offset:offset + length],
                "total": len(hits),
                "features": scan["features"],
                "engine": "local-scan",
            }
        except HTTPException as exc:
            logger.info("local scan failed (%s) -> /search", exc.detail)

    # ---- 3) HF full-text /search ----------------------------------------
    try:
        data = await hf_search(config, split, query, offset, length)
        records = normalise_records(data.get("rows", []))
        total = data.get("num_rows_total")
        return {
            "records": records,
            "total": int(total if total is not None else len(records)),
            "features": data.get("features", features) or features,
            "engine": "hf-search",
        }
    except HTTPException as exc:
        logger.info("search failed (%s)", exc.detail)
        raise HTTPException(
            status_code=502,
            detail="Search backend is temporarily busy. Please retry.",
        )


# ------------------------------------------------------- device / location ---

UA_MODEL_PATTERNS = [
    r"(SM-[A-Z0-9]+)",
    r"(SM[A-Z0-9]{3,})",
    r"\b(CPH\d{4,})\b",
    r"\b(Redmi[ A-Za-z0-9.]*)",
    r"\b(POCO[ A-Za-z0-9.]*)",
    r"\b(M[0-9]{4}[A-Z0-9]*)\b",
    r"\b(V[0-9]{4}[A-Z0-9]*)\b",
    r"\b(RMX\d{4,})\b",
    r"\b(Pixel [0-9a-zA-Z ]+)",
    r"\b(OnePlus[ A-Za-z0-9]*)",
    r"\b(Moto [A-Za-z0-9]+)",
    r"\b(iPhone[0-9,]+)",
    r"\b(iPad[0-9,]+)",
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
            name = "Opera" if token == "OPR" else ("Edge" if token == "Edg" else token)
            if name == "Version":
                name = "Safari"
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
    """Vercel headers first, warna ip-api.com se city/ISP/mobile."""
    if ip in _GEO_CACHE:
        stamp, value = _GEO_CACHE[ip]
        if time.time() - stamp < 3600:
            return value

    result = {}
    if LOCATION_LOOKUP and ip and not ip.startswith(("127.", "10.", "192.168.", "::1")):
        try:
            response = await asyncio.wait_for(
                app.state.client.get(
                    f"http://ip-api.com/json/{ip}",
                    params={
                        "fields": "status,country,countryCode,regionName,city,"
                                  "isp,org,mobile,query"
                    },
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
    city = request.headers.get("x-vercel-ip-city") or ""
    region = request.headers.get("x-vercel-ip-country-region") or ""
    country = request.headers.get("x-vercel-ip-country") or ""
    return {"city": city, "region": region, "country": country, "org": ""}


# ------------------------------------------------------------ bot notify ----

def build_bot_message(
    *,
    key: str,
    query: str,
    matches: int,
    returned: int,
    limit: int,
    via: str,
    request: Request,
    ip: str,
    geo: dict,
) -> str:
    ua = request.headers.get("user-agent", "")
    ua_short = ua if len(ua) <= 96 else ua[:96] + "…"
    location = ", ".join([p for p in (geo.get("city"), geo.get("region"), geo.get("country")) if p])
    org = geo.get("org") or ""
    if org:
        location = f"{location} · {org}" if location else org

    device_fp = hashlib.sha256(
        f"{ua}|{ip}|{request.headers.get('accept-language', '')}".encode("utf-8")
    ).hexdigest()[:16]

    query_line = f'q -> any field (partial) = "{query}"' if query else "q -> none (browse mode)"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"🔎 SEARCH — {API_TITLE}",
        "",
        f"👤 Key: {mask_key(key)} (id {short_id(key)})",
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
    ]
    return "\n".join(lines)


async def send_bot(text: str) -> None:
    if not BOT_ENABLED:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        # Telegram limit 4096 -> chunks
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
        if not BOT_SILENT:
            pass


async def notify(**kwargs) -> None:
    try:
        text = build_bot_message(**kwargs)
        await send_bot(text)
    except Exception:
        logger.exception("notification build failed")


# ---------------------------------------------------------------- payload ---

def build_response(
    *,
    key: str,
    config: str,
    split: str,
    records: list,
    total: int,
    offset: int,
    limit: int,
    query: str,
    features: list,
    engine: str,
) -> dict:
    count = len(records)
    base = f"/{key}" + (f"/{query}" if query else "")

    def page_url(target_offset: int) -> str:
        page_no = (target_offset // limit) + 1 if limit else 1
        return f"{base}?page={page_no}&limit={limit}"

    has_more = offset + count < (total or 0)
    return {
        "success": True,
        "key": key,
        "dataset": HF_DATASET,
        "config": config,
        "split": split,
        "query": query or None,
        "engine": engine,
        "total": int(total or 0),
        "count": count,
        "offset": offset,
        "limit": limit,
        "page": (offset // limit) + 1 if limit else 1,
        "has_more": has_more,
        "next": page_url(offset + limit) if has_more else None,
        "prev": page_url(max(0, offset - limit)) if offset > 0 else None,
        "fields": feature_names(features),
        "data": records,
    }


async def serve(
    request: Request,
    key: str,
    query: str,
    page: int,
    limit: int,
    offset: Optional[int],
    config: Optional[str],
    split: Optional[str],
    notify_flag: bool,
    via: str,
) -> JSONResponse:
    # ---- key validate -----------------------------------------------------
    key = (key or "").strip()
    if not key or len(key) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", key):
        raise HTTPException(status_code=400, detail="Invalid key format.")
    if ALLOWED_KEYS and key not in ALLOWED_KEYS:
        raise HTTPException(status_code=403, detail="This key is not allowed.")

    # ---- pagination -------------------------------------------------------
    limit = max(1, min(MAX_PAGE_SIZE, int(limit or PAGE_SIZE)))
    if offset is None:
        page = max(1, int(page or 1))
        offset = (page - 1) * limit
    offset = max(0, int(offset))

    use_config = (config or HF_CONFIG).strip() or HF_CONFIG
    use_split = (split or HF_SPLIT).strip() or HF_SPLIT
    resolved = await resolve_split(use_config, use_split)
    use_config = resolved.get("config", use_config)
    use_split = resolved.get("split", use_split)

    query = (query or "").strip()

    # ---- data -------------------------------------------------------------
    if query:
        result = await run_search(use_config, use_split, query, offset, limit)
        records = result["records"]
        total = result["total"]
        features = result["features"]
        engine = result["engine"]
    else:
        data = await fetch_rows(use_config, use_split, offset, limit)
        records = normalise_records(data.get("rows", []), offset)
        total = int(data.get("num_rows_total") or 0)
        features = data.get("features", []) or []
        engine = "hf-rows"

    payload = build_response(
        key=key,
        config=use_config,
        split=use_split,
        records=records,
        total=total,
        offset=offset,
        limit=limit,
        query=query,
        features=features,
        engine=engine,
    )

    # ---- owner notification ----------------------------------------------
    if notify_flag:
        ip = client_ip(request)
        geo = vercel_location(request)
        if not (geo.get("city") or geo.get("country")):
            geo = await lookup_location(ip)
        else:
            extra = await lookup_location(ip)
            if extra.get("org"):
                geo["org"] = extra["org"]
        await notify(
            key=key,
            query=query,
            matches=payload["total"],
            returned=payload["count"],
            limit=limit,
            via=via,
            request=request,
            ip=ip,
            geo=geo,
        )

    return JSONResponse(payload)


# ------------------------------------------------------------------ routes ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset_configured": bool(HF_DATASET),
        "dataset": HF_DATASET or None,
        "bot_enabled": BOT_ENABLED,
        "keys_restricted": bool(ALLOWED_KEYS),
        "provider": "Hugging Face Dataset Viewer API",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/dataset")
async def dataset_info():
    result = await hf_get("splits", {})
    splits = result.get("splits", [])
    return {
        "success": True,
        "dataset": HF_DATASET,
        "split_count": len(splits),
        "splits": splits,
    }


@app.get("/api/fields")
async def dataset_fields(
    config: str = Query(default=None),
    split: str = Query(default=None),
):
    resolved = await resolve_split((config or HF_CONFIG), (split or HF_SPLIT))
    data = await fetch_rows(resolved["config"], resolved["split"], 0, 1)
    return {
        "success": True,
        "config": resolved["config"],
        "split": resolved["split"],
        "total": data.get("num_rows_total"),
        "fields": data.get("features", []),
    }


@app.get("/api/v1/{key}")
async def api_browse(
    request: Request,
    key: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=100),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = Query(default=None),
    split: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=300),
    notify: int = Query(default=1, ge=0, le=1),
):
    return await serve(request, key, q or "", page, limit, offset, config, split,
                       bool(notify), "api URL (/api/v1/key)")


@app.get("/api/v1/{key}/{query}")
async def api_search(
    request: Request,
    key: str,
    query: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=100),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = Query(default=None),
    split: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=300),
    notify: int = Query(default=1, ge=0, le=1),
):
    return await serve(request, key, query or q or "", page, limit, offset, config, split,
                       bool(notify), "api URL (/api/v1/key/query)")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/robots.txt", include_in_schema=False)
async def robots():
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HOME_HTML)


@app.get("/{key}")
async def path_browse(
    request: Request,
    key: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=100),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = Query(default=None),
    split: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=300),
    notify: int = Query(default=1, ge=0, le=1),
):
    return await serve(request, key, q or "", page, limit, offset, config, split,
                       bool(notify), "path-key URL (/key)")


@app.get("/{key}/{query}")
async def path_search(
    request: Request,
    key: str,
    query: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE, ge=1, le=100),
    offset: Optional[int] = Query(default=None, ge=0),
    config: Optional[str] = Query(default=None),
    split: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=300),
    notify: int = Query(default=1, ge=0, le=1),
):
    return await serve(request, key, query or q or "", page, limit, offset, config, split,
                       bool(notify), "path-key URL (/key/query)")


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "status": exc.status_code},
    )


# ------------------------------------------------------------------- HTML ---

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset JSON API</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px 16px 80px;
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    background: #0b0f1a; color: #e6e9f2;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b93a7; font-size: 13px; margin-bottom: 18px; }
  .card {
    background: #121829; border: 1px solid #1f2740; border-radius: 14px;
    padding: 16px; margin-bottom: 14px;
  }
  label { display: block; font-size: 12px; color: #8b93a7; margin-bottom: 6px; }
  input, select, button {
    font: inherit; border-radius: 10px; border: 1px solid #26304a;
    background: #0d1424; color: #e6e9f2; padding: 10px 12px; outline: none;
  }
  input:focus, select:focus { border-color: #3b82f6; }
  button { background: #2b6ef6; border-color: #2b6ef6; color: #fff; cursor: pointer; font-weight: 600; }
  button:disabled { opacity: .5; cursor: default; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }
  .row > div { flex: 1 1 180px; }
  .urls { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  .urls a { color: #7fb0ff; word-break: break-all; }
  pre {
    background: #0d1424; border: 1px solid #1f2740; border-radius: 12px;
    padding: 14px; overflow: auto; max-height: 60vh; font-size: 12.5px;
  }
  .item {
    border-bottom: 1px solid #1b2237; padding: 10px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
    white-space: pre-wrap; word-break: break-word;
  }
  .item:last-child { border-bottom: 0; }
  .idx { color: #5c6a8a; }
  .muted { color: #8b93a7; font-size: 13px; }
  .badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
           background: #1b2740; color: #9fb4dd; margin-left: 6px; }
  .loader { text-align: center; color: #8b93a7; padding: 14px; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Dataset JSON API <span class="badge">infinite scroll</span></h1>
  <div class="sub">Har response JSON me aata hai — keys ke sath. Scroll karo, data apne aap load hota rahega.</div>

  <div class="card">
    <div class="row">
      <div>
        <label>Your key (URL segment)</label>
        <input id="key" placeholder="e.g. 14ec092a" autocomplete="off">
      </div>
      <div>
        <label>Query (optional)</label>
        <input id="q" placeholder="koi bhi naam / value — sabhi fields me dhoondhega" autocomplete="off">
      </div>
      <div style="flex:0 0 auto">
        <button id="go">Load</button>
      </div>
    </div>
    <div class="urls" id="urls" style="margin-top:12px"></div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <div style="flex:0 0 auto"><label>Limit</label>
        <select id="limit">
          <option>20</option><option>50</option><option>100</option>
        </select>
      </div>
      <div style="flex:0 0 auto"><label>View</label>
        <select id="view"><option value="list">List</option><option value="json">Raw JSON</option></select>
      </div>
      <div style="flex:0 0 auto"><button id="reset" class="ghost">Reset</button></div>
    </div>
    <div id="status" class="muted">Key daalo aur Load dabao.</div>
    <div id="out"></div>
    <div id="sentinel" class="loader"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const state = { key: "", q: "", limit: 20, offset: 0, total: 0, loading: false, done: true, all: [] };

function apiUrl(page, limit) {
  const base = "/" + encodeURIComponent(state.key) +
               (state.q ? "/" + encodeURIComponent(state.q) : "");
  return base + "?page=" + page + "&limit=" + limit;
}

function renderUrls() {
  if (!state.key) { $("urls").textContent = ""; return; }
  const o = location.origin;
  const browse = o + "/" + encodeURIComponent(state.key);
  const search = state.q ? o + "/" + encodeURIComponent(state.key) + "/" + encodeURIComponent(state.q) : null;
  $("urls").innerHTML =
    "Browse: <a href='" + browse + "' target='_blank'>" + browse + "</a>" +
    (search ? "<br>Search: <a href='" + search + "' target='_blank'>" + search + "</a>" : "");
}

function rowHtml(rec) {
  const copy = Object.assign({}, rec);
  const idx = copy._index;
  delete copy._index;
  return "<div class='item'>" + (idx !== undefined ? "<span class='idx'>#" + idx + "</span>  " : "") +
         escapeHtml(JSON.stringify(copy, null, 0)) + "</div>";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderAll() {
  if ($("view").value === "json") {
    $("out").innerHTML = "<pre>" + escapeHtml(JSON.stringify(state.all, null, 2)) + "</pre>";
  } else {
    $("out").innerHTML = state.all.length
      ? state.all.map(rowHtml).join("")
      : "<div class='muted'>Kuch nahi mila.</div>";
  }
}

async function load(reset) {
  if (!state.key) { $("status").textContent = "Pehle key daalo."; return; }
  if (state.loading) return;
  if (!reset && state.done) return;

  state.loading = true;
  const page = reset ? 1 : Math.floor(state.offset / state.limit) + 1;
  $("sentinel").textContent = "Loading page " + page + "…";

  try {
    const res = await fetch(apiUrl(page, state.limit));
    const data = await res.json();
    if (!res.ok || data.success === false) throw new Error(data.error || data.detail || "Request failed");

    if (reset) { state.all = []; state.offset = 0; }
    state.all = state.all.concat(data.data || []);
    state.total = data.total || 0;
    state.offset = state.all.length;
    state.done = !data.has_more;

    $("status").innerHTML = "Total <b>" + state.total + "</b> · loaded <b>" + state.offset +
      "</b> · engine <b>" + (data.engine || "-") + "</b>" + (data.query ? " · query <b>" + escapeHtml(data.query) + "</b>" : "");
    renderAll();
    $("sentinel").textContent = state.done ? (state.all.length ? "— end —" : "") : "";
    renderUrls();
  } catch (err) {
    $("sentinel").textContent = "";
    $("status").textContent = "Error: " + err.message;
  } finally {
    state.loading = false;
  }
}

$("go").onclick = () => {
  state.key = $("key").value.trim();
  state.q = $("q").value.trim();
  state.limit = parseInt($("limit").value, 10);
  state.offset = 0; state.done = false;
  load(true);
};
$("q").addEventListener("keydown", e => { if (e.key === "Enter") $("go").click(); });
$("key").addEventListener("keydown", e => { if (e.key === "Enter") $("go").click(); });
$("limit").onchange = () => { state.limit = parseInt($("limit").value, 10); state.done = false; load(true); };
$("view").onchange = renderAll;
$("reset").onclick = () => {
  state.all = []; state.offset = 0; state.done = true; state.key = ""; state.q = "";
  $("out").innerHTML = ""; $("status").textContent = "Reset ho gaya.";
};

new IntersectionObserver(entries => {
  if (entries[0].isIntersecting && !state.done) load(false);
}, { rootMargin: "320px" }).observe($("sentinel"));

// URL /key ya /key/query se seedha prefill
const parts = location.pathname.split("/").filter(Boolean);
if (parts.length >= 1) { $("key").value = decodeURIComponent(parts[0]); state.key = $("key").value; }
if (parts.length >= 2) { $("q").value = decodeURIComponent(parts[1]); state.q = $("q").value; }
if (state.key) { state.limit = 20; state.done = false; load(true); }
</script>
</body>
</html>
"""


# Vercel Python runtime entry point: `app` is auto-detected.
