#!/usr/bin/env bash
echo "停止 Img2CAD 服务..."
pkill -f "cpolar"          2>/dev/null && echo "  cpolar 已停止" || echo "  cpolar 未运行"
pkill -f "server/main.py"  2>/dev/null && echo "  服务器已停止" || echo "  服务器未运行"
pkill -f "uvicorn"         2>/dev/null || true
echo "完成"
