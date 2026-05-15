# Img2CAD 服务启动指南

## 一键启动

```bash
cd /root/autodl-tmp/Img2CAD && bash start.sh
```

等待约 30 秒（torch 加载较慢），然后浏览器打开固定域名：

> **http://img2cad.cpolar.cn**

---

## 固定公网地址

| 协议 | 地址 |
|------|------|
| HTTP | http://img2cad.cpolar.cn |
| HTTPS | https://img2cad.cpolar.cn |

> 已通过 cpolar VIP（中国区）预留子域名 `img2cad`，每次启动地址不变。

---

## 手动启动（分步操作）

### 1. 启动 Img2CAD 服务器

```bash
source /root/autodl-tmp/Img2CAD/img2cad_env/bin/activate
python /root/autodl-tmp/Img2CAD/server/main.py
```

服务器监听 `http://0.0.0.0:8000`，API 文档见 `/docs`。

### 2. 新开终端，启动 cpolar 内网穿透

```bash
cpolar http 8000 -region=cn -subdomain=img2cad
```

---

## Cpolar 管理命令

```bash
cpolar tunnel list                # 查看隧道列表
ps aux | grep cpolar              # 查看运行状态
grep "Tunnel established" /var/log/cpolar/access.log | tail -2  # 查看隧道地址
```

---

## 停止服务

```bash
bash /root/autodl-tmp/Img2CAD/stop.sh
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| GPU 不可用 | `nvidia-smi` 确认驱动已加载 |
| HTTPS 证书错误 | 先用 HTTP 访问，等几分钟后证书自动签发 |
| 端口 8000 被占用 | `lsof -i :8000` 查看并 `kill` |
| cpolar 未登录 | `cpolar authtoken <token>` 重新认证 |
| 域名不通 | 确认 cpolar 和 server 进程都在运行 |
