#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已生成 .env，请按需修改后重新运行。"
fi

# shellcheck disable=SC1091
source .env 2>/dev/null || true

GATEWAY_PORT="${DOTS_GATEWAY_PORT:-8030}"
RAG_PORT="${RAG_API_PORT:-8020}"

start_gateway() {
  cd "$ROOT/services/dots_ocr_gateway"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
  fi
  echo "启动 Dots.OCR 解析网关 :$GATEWAY_PORT"
  .venv/bin/uvicorn main:app --host 0.0.0.0 --port "$GATEWAY_PORT" &
  echo $! > "$ROOT/.dots_gateway.pid"
  cd "$ROOT"
}

start_rag() {
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
  fi
  echo "启动 RAG API :$RAG_PORT"
  .venv/bin/uvicorn app.api:app --host 0.0.0.0 --port "$RAG_PORT" &
  echo $! > "$ROOT/.rag_api.pid"
}

stop_all() {
  for pidfile in .dots_gateway.pid .rag_api.pid; do
    if [[ -f $pidfile ]]; then
      kill "$(cat "$pidfile")" 2>/dev/null || true
      rm -f "$pidfile"
    fi
  done
}

case "${1:-start}" in
  start)
    stop_all
    start_gateway
    sleep 1
    start_rag
    echo "Gateway: http://127.0.0.1:$GATEWAY_PORT/health"
    echo "RAG API: http://127.0.0.1:$RAG_PORT/health"
    ;;
  stop)
    stop_all
    echo "已停止本地服务"
    ;;
  *)
    echo "用法: $0 [start|stop]"
    exit 1
    ;;
esac
