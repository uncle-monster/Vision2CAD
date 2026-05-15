#!/usr/bin/env bash
set -e

PROJECT_DIR="/root/autodl-tmp/Img2CAD"
VENV="$PROJECT_DIR/img2cad_env/bin/activate"
CPOLAR_REGION="cn"
CPOLAR_SUBDOMAIN="img2cad"
PUBLIC_URL="http://${CPOLAR_SUBDOMAIN}.cpolar.cn"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Img2CAD 服务一键启动${NC}"
echo -e "${GREEN}============================================${NC}"

# ---- Step 1: Kill old processes ----
echo -e "${CYAN}[1/4] 清理旧进程...${NC}"
kill $(pgrep -f "cpolar" 2>/dev/null) 2>/dev/null || true
kill $(pgrep -f "server/main.py" 2>/dev/null) 2>/dev/null || true
kill $(pgrep -f "uvicorn" 2>/dev/null) 2>/dev/null || true
sleep 1
echo "  完成"

# ---- Step 2: Start cpolar ----
echo -e "${CYAN}[2/4] 启动 cpolar 隧道 (域名: ${PUBLIC_URL})...${NC}"
nohup cpolar http 8000 \
    -region="$CPOLAR_REGION" \
    -subdomain="$CPOLAR_SUBDOMAIN" \
    --log=/var/log/cpolar/access.log \
    > /tmp/cpolar_stdout.log 2>&1 &
echo "  cpolar PID: $!"
sleep 5

# Verify tunnel
if grep -q "Tunnel established" /var/log/cpolar/access.log 2>/dev/null; then
    echo -e "  ${GREEN}隧道已建立${NC}"
else
    echo -e "  ${RED}隧道建立失败，查看日志: tail /var/log/cpolar/access.log${NC}"
fi

# ---- Step 3: Start server ----
echo -e "${CYAN}[3/4] 启动 Img2CAD 服务器 (加载模型约需 30 秒)...${NC}"
source "$VENV"
nohup python "$PROJECT_DIR/server/main.py" > /tmp/img2cad_server.log 2>&1 &
echo "  服务器 PID: $!"

# ---- Step 4: Wait for ready ----
echo -e "${CYAN}[4/4] 等待服务器就绪...${NC}"
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}服务器就绪 (耗时 ${i}s)${NC}"
        break
    fi
    printf "."
    sleep 1
done
echo ""

# ---- Done ----
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   启动完成！${NC}"
echo -e "${GREEN}   公网地址: ${PUBLIC_URL}${NC}"
echo -e "${GREEN}   API 文档: ${PUBLIC_URL}/docs${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "查看日志:"
echo "  服务器: tail -f /tmp/img2cad_server.log"
echo "  cpolar: tail -f /var/log/cpolar/access.log"
echo ""
echo "停止服务: bash ${PROJECT_DIR}/stop.sh"
