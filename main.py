
import os
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-dataset-api")

HF_DATASET = os.getenv("HF_DATASET", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_API = "https://datasets-server.huggingface.co"
TIMEOUT = httpx.Timeout(20.0, connect=8.0)


def get_headers():
    headers = {"User-Agent": "HF-Dataset-Explorer/1.0"}

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    return headers


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
    title="Hugging Face Dataset API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Hugging Face request helper
# --------------------------------------------------

async def hf_get(endpoint: str, params: dict):
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

        if response.status_code >= 400:
            logger.error(
                "Hugging Face returned HTTP %s for %s",
                response.status_code,
                endpoint,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Hugging Face API returned HTTP "
                    f"{response.status_code}."
                ),
            )

        data = response.json()

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="Unexpected response from Hugging Face.",
            )

        if data.get("error"):
            logger.warning(
                "Hugging Face API reported a dataset error: %s",
                data["error"],
            )
            raise HTTPException(
                status_code=502,
                detail="Hugging Face could not serve this dataset.",
            )

        return data

    except HTTPException:
        raise

    except httpx.TimeoutException:
        logger.exception("Hugging Face request timed out")
        raise HTTPException(
            status_code=504,
            detail="Hugging Face request timed out. Try again.",
        )

    except httpx.RequestError:
        logger.exception("Hugging Face network request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Hugging Face.",
        )

    except ValueError:
        logger.exception("Invalid JSON from Hugging Face")
        raise HTTPException(
            status_code=502,
            detail="Hugging Face returned invalid JSON.",
        )

    except Exception:
        logger.exception("Unexpected dataset API error")
        raise HTTPException(
            status_code=500,
            detail="Unexpected internal error.",
        )


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset_configured": bool(HF_DATASET),
        "provider": "Hugging Face Dataset Viewer API",
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


@app.get("/api/rows")
async def dataset_rows(
    config: str = Query(default="default", min_length=1, max_length=200),
    split: str = Query(default="train", min_length=1, max_length=100),
    offset: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=100),
):
    # Verify the selected config and split before requesting rows.
    metadata = await hf_get("splits", {})

    available = metadata.get("splits", [])

    valid = any(
        item.get("config") == config
        and item.get("split") == split
        for item in available
    )

    if not valid:
        raise HTTPException(
            status_code=422,
            detail="Invalid config or split. Check /api/dataset.",
        )

    result = await hf_get(
        "rows",
        {
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        },
    )

    return {
        "success": True,
        "dataset": HF_DATASET,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
        "total": result.get("num_rows_total"),
        "features": result.get("features", []),
        "rows": result.get("rows", []),
    }


@app.get("/api/search")
async def search_dataset(
    q: str = Query(min_length=1, max_length=200),
    config: str = Query(default="default"),
    split: str = Query(default="train"),
):
    metadata = await hf_get("splits", {})

    valid = any(
        item.get("config") == config
        and item.get("split") == split
        for item in metadata.get("splits", [])
    )

    if not valid:
        raise HTTPException(
            status_code=422,
            detail="Invalid config or split.",
        )

    return await hf_get(
        "search",
        {
            "config": config,
            "split": split,
            "query": q,
        },
    )


# --------------------------------------------------
# Simple browser UI
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HF Dataset Explorer</title>
<style>
body {
  font: 16px Arial, sans-serif;
  margin: 24px auto;
  padding: 0 16px;
  max-width: 1000px;
  background: #f5f6fa;
  color: #202536;
}
section {
  background: white;
  padding: 18px;
  margin: 14px 0;
  border-radius: 12px;
}
button, select, input {
  padding: 10px;
  margin: 5px 3px 5px 0;
}
button { cursor: pointer; }
#table { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td {
  text-align: left;
  padding: 10px;
  border-bottom: 1px solid #ddd;
  overflow-wrap: anywhere;
}
th { background: #edf0f6; }
</style>
</head>
<body>
<h1>Hugging Face Dataset Explorer</h1>

<section>
  <p id="status">Loading dataset metadata...</p>
  <label>Config: <select id="config"></select></label>
  <label>Split: <select id="split"></select></label>
  <button id="load">Load records</button>
</section>

<section>
  <input id="query" placeholder="Search dataset text">
  <button id="search">Search</button>
  <div id="table">No data loaded.</div>
  <p>
    <button id="prev" disabled>Previous</button>
    <span id="page">Page 1</span>
    <button id="next" disabled>Next</button>
  </p>
</section>

<script>
const $ = id => document.getElementById(id);
let offset = 0;
const length = 20;
let total = null;
let splits = [];

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

function selectedSplit() {
  return JSON.parse($("split").value);
}

function displayRows(rows) {
  if (!rows.length) {
    $("table").textContent = "No records found.";
    return;
  }

  const columns = Object.keys(rows[0].row || {});
  const table = document.createElement("table");
  const head = document.createElement("tr");

  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col;
    head.appendChild(th);
  }

  const thead = document.createElement("thead");
  thead.appendChild(head);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  for (const item of rows) {
    const tr = document.createElement("tr");

    for (const col of columns) {
      const td = document.createElement("td");
      const value = item.row[col];

      td.textContent =
        value === null || value === undefined
          ? ""
          : typeof value === "object"
            ? JSON.stringify(value)
            : String(value);

      tr.appendChild(td);
    }

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  $("table").replaceChildren(table);
}

async function loadRows() {
  $("status").textContent = "Loading records...";
  $("next").disabled = true;
  $("prev").disabled = true;

  try {
    const s = selectedSplit();
    const params = new URLSearchParams({
      config: s.config,
      split: s.split,
      offset: String(offset),
      length: String(length)
    });

    const data = await api("/api/rows?" + params);

    displayRows(data.rows || []);
    total = data.total ?? null;

    $("status").textContent =
      "Loaded " + (data.rows || []).length + " records.";

    $("page").textContent =
      "Page " + (Math.floor(offset / length) + 1) +
      (total !== null ? " · Total: " + total : "");

    $("prev").disabled = offset === 0;
    $("next").disabled =
      total !== null && offset + length >= total;

  } catch (e) {
    $("status").textContent = "Error: " + e.message;
    $("table").textContent = "Could not load records.";
  }
}

async function searchRows() {
  const q = $("query").value.trim();

  if (!q) {
    $("status").textContent = "Enter a search term.";
    return;
  }

  try {
    const s = selectedSplit();
    const params = new URLSearchParams({
      q,
      config: s.config,
      split: s.split
    });

    const data = await api("/api/search?" + params);
    displayRows(data.rows || []);
    $("status").textContent = "Search completed.";
  } catch (e) {
    $("status").textContent = "Search error: " + e.message;
  }
}

async function init() {
  try {
    const data = await api("/api/dataset");
    splits = data.splits || [];

    if (!splits.length) {
      throw new Error("No available dataset splits.");
    }

    const configs = [...new Set(splits.map(x => x.config))];
    $("config").replaceChildren();

    for (const config of configs) {
      const opt = document.createElement("option");
      opt.value = config;
      opt.textContent = config;
      $("config").appendChild(opt);
    }

    function updateSplits() {
      const config = $("config").value;
      const matches = splits.filter(x => x.config === config);

      $("split").replaceChildren();

      for (const item of matches) {
        const opt = document.createElement("option");
        opt.value = JSON.stringify(item);
        opt.textContent = item.split;
        $("split").appendChild(opt);
      }

      offset = 0;
    }

    $("config").addEventListener("change", () => {
      updateSplits();
      loadRows();
    });

    $("split").addEventListener("change", () => {
      offset = 0;
      loadRows();
    });

    updateSplits();
    await loadRows();

  } catch (e) {
    $("status").textContent = "Dataset error: " + e.message;
  }
}

$("load").onclick = () => loadRows();
$("search").onclick = () => searchRows();

$("prev").onclick = () => {
  offset = Math.max(0, offset - length);
  loadRows();
};

$("next").onclick = () => {
  if (total === null || offset + length < total) {
    offset += length;
    loadRows();
  }
};

init();
</script>
</body>
</html>
""")


# Vercel Python runtime entry point
# Export the FastAPI application as `app`.
