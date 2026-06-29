from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.diagnosis import router as diagnosis_router
from app.api.evaluation import router as evaluation_router
from app.api.health import router as health_router
from app.api.manual import router as manual_router
from app.api.trace import router as trace_router
from app.core.config import settings


LANDING_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ManualMind-Agent</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      background: #f7f9fc;
    }
    main {
      max-width: 920px;
      margin: 0 auto;
      padding: 56px 24px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 42px;
      line-height: 1.1;
    }
    .lead {
      font-size: 20px;
      color: #334155;
      margin: 0 0 20px;
    }
    .notice {
      padding: 14px 16px;
      border: 1px solid #d9e2ef;
      border-radius: 8px;
      background: #ffffff;
      color: #475569;
    }
    h2 {
      margin-top: 32px;
      font-size: 22px;
    }
    ul {
      line-height: 1.8;
      padding-left: 22px;
    }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 28px;
    }
    a {
      color: #155eef;
    }
    .button {
      display: inline-block;
      padding: 10px 14px;
      border: 1px solid #bfd0ff;
      border-radius: 8px;
      background: #ffffff;
      text-decoration: none;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <main>
    <h1>ManualMind-Agent</h1>
    <p class="lead">复杂设备手册问答与故障诊断多 Agent Demo</p>
    <p class="notice">
      This deployment uses synthetic demo data only. It does not include real
      manufacturer manuals, private industrial documents, or production customer data.
    </p>

    <h2>Project Highlights</h2>
    <ul>
      <li>Multi-agent diagnosis workflow</li>
      <li>Hybrid retrieval</li>
      <li>Streaming safety guard</li>
      <li>Text-based PDF ingestion</li>
      <li>Optional LLM report fallback</li>
      <li>Optional Milvus vector backend</li>
      <li>Trace / Evaluation / Human handoff</li>
    </ul>

    <div class="links" aria-label="Project links">
      <a class="button" href="/docs">API Docs</a>
      <a class="button" href="/api/health">Health Check</a>
      <a class="button" href="/openapi.json">OpenAPI JSON</a>
      <a class="button" href="https://github.com/Xiejiahao0413/ManualMind-Agent">GitHub Repository</a>
    </div>
  </main>
</body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.get("/", response_class=HTMLResponse)
    def landing_page() -> HTMLResponse:
        return HTMLResponse(content=LANDING_PAGE_HTML)

    app.include_router(health_router, prefix="/api")
    app.include_router(diagnosis_router, prefix="/api")
    app.include_router(manual_router, prefix="/api")
    app.include_router(trace_router, prefix="/api")
    app.include_router(evaluation_router, prefix="/api")
    return app


app = create_app()
