# CryptoAgent — Cloud VPS Hosting & Deployment Plan

> Goal: Move from local Windows machine to a 24/7 cloud VPS with secure dashboard access,
> auto-restart on crash, log rotation, and a weekly retraining workflow that stays on your
> local GPU machine.

---

## 1. VPS Selection

### Recommended Specs
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| vCPU     | 2       | 4           |
| RAM      | 4 GB    | 8 GB        |
| SSD      | 40 GB   | 80 GB       |
| Network  | 1 Gbps  | 1 Gbps      |
| OS       | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

**Memory breakdown (live agent):**
- Agent process: ~1.2 GB (60 live feeds × 500-bar buffers × 75 features)
- Dashboard: ~300 MB
- TimescaleDB: ~300 MB
- Redis: ~50 MB
- OS overhead: ~500 MB
- **Total: ~2.4 GB** → 4 GB minimum, 8 GB comfortable

### Provider Recommendations (cheapest first)

| Provider | Plan | vCPU | RAM | SSD | Price/mo | Region |
|----------|------|------|-----|-----|----------|--------|
| **Hetzner** | CPX21 | 3 | 4 GB | 80 GB | ~$7 | Singapore |
| **Hetzner** | CPX31 | 4 | 8 GB | 160 GB | ~$17 | Singapore |
| **DigitalOcean** | Droplet | 2 | 4 GB | 80 GB | $24 | Singapore |
| **Vultr** | Cloud Compute | 4 | 8 GB | 160 GB | $40 | Tokyo |
| **AWS Lightsail** | Bundle | 2 | 4 GB | 80 GB | $20 | ap-southeast-1 |

**Recommendation: Hetzner CPX31 (Singapore)** — best value, low latency to BloFin (Singapore-based), solid uptime SLA.

> GPU is NOT needed on VPS. Run weekly LGBM retraining on your local RTX 5070,
> then rsync the updated model files to the VPS. LSTM training also stays local.

---

## 2. Initial Server Setup

```bash
# 1. SSH in as root
ssh root@<VPS_IP>

# 2. Create non-root user
adduser cryptoagent
usermod -aG sudo cryptoagent

# 3. Copy your SSH key to the new user
rsync --archive --chown=cryptoagent:cryptoagent ~/.ssh /home/cryptoagent

# 4. Switch to new user for all remaining steps
su - cryptoagent

# 5. Update system
sudo apt update && sudo apt upgrade -y

# 6. Install essentials
sudo apt install -y git curl wget htop ufw python3.12 python3.12-venv \
    python3.12-dev build-essential libffi-dev libssl-dev

# 7. Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker cryptoagent
newgrp docker   # refresh group without logout

# 8. Install Docker Compose v2
sudo apt install -y docker-compose-plugin
docker compose version   # should print v2.x
```

---

## 3. Firewall

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp        # SSH
sudo ufw allow 80/tcp        # HTTP (Nginx redirect)
sudo ufw allow 443/tcp       # HTTPS (dashboard)
sudo ufw enable
sudo ufw status
```

> Do NOT open port 8501 (Streamlit) or 5433 (TimescaleDB) to the internet.
> Dashboard is served via Nginx reverse proxy on 443 only.

---

## 4. Project Deployment

### 4a. Clone repo

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/cryptoagent.git
cd cryptoagent
```

Or if private repo:
```bash
git clone git@github.com:YOUR_USERNAME/cryptoagent.git
```

### 4b. Upload secrets

From your **local machine** (never commit `.env`):

```bash
# Windows (using scp or rsync via WSL/Git Bash)
scp C:/Users/ZESTRO/Desktop/cryptoagent/.env cryptoagent@<VPS_IP>:~/cryptoagent/.env
```

Or create manually on VPS:
```bash
nano ~/cryptoagent/.env
# Paste your BLOFIN_API_KEY, BLOFIN_API_SECRET, BLOFIN_PASSPHRASE,
# TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY
```

### 4c. Upload trained models

From your local machine:
```bash
# Windows Git Bash / WSL
rsync -avz --progress \
    /c/Users/ZESTRO/Desktop/cryptoagent/models/ \
    cryptoagent@<VPS_IP>:~/cryptoagent/models/
```

This copies all 40 LGBM + 10 LSTM models (~500 MB).

### 4d. Create required directories

```bash
mkdir -p ~/cryptoagent/logs ~/cryptoagent/backtests
```

---

## 5. Docker Compose — Production Config

The existing `docker-compose.yml` works as-is for the agent + DB + Redis.
Make one tweak: remove the exposed TimescaleDB port from external access.

Edit `docker-compose.yml` on the VPS:
```bash
nano ~/cryptoagent/docker-compose.yml
```

Change the timescaledb ports section from:
```yaml
ports:
  - "5433:5432"
```
To (localhost only):
```yaml
ports:
  - "127.0.0.1:5433:5432"
```

Also change Redis ports:
```yaml
ports:
  - "127.0.0.1:6379:6379"
```

And remove the `agent:` and `dashboard:` services from docker-compose — run those as separate systemd services (section 7) so they can be managed and restarted independently.

### Start infrastructure only

```bash
cd ~/cryptoagent
docker compose up -d timescaledb redis
docker compose ps   # both should show "healthy"
```

---

## 6. Nginx + HTTPS Dashboard

### Install Nginx + Certbot

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### Point a domain at the VPS

In your domain registrar's DNS, add an A record:
```
dashboard.yourdomain.com  →  <VPS_IP>
```

### Nginx config

```bash
sudo nano /etc/nginx/sites-available/cryptoagent
```

Paste:
```nginx
server {
    listen 80;
    server_name dashboard.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name dashboard.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/dashboard.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/dashboard.yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # Basic auth (prevents public access)
    auth_basic           "CryptoAgent";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }
}
```

### Enable site + get SSL cert

```bash
sudo ln -s /etc/nginx/sites-available/cryptoagent /etc/nginx/sites-enabled/
sudo nginx -t

# Create password for dashboard basic auth
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin
# Enter a password when prompted

# Get SSL cert
sudo certbot --nginx -d dashboard.yourdomain.com
# Certbot auto-renews via systemd timer (check: systemctl list-timers | grep certbot)

sudo systemctl reload nginx
```

Dashboard is now at `https://dashboard.yourdomain.com` (no ngrok needed).

---

## 7. Systemd Services (auto-restart on crash/reboot)

### 7a. Agent service

```bash
sudo nano /etc/systemd/system/cryptoagent.service
```

```ini
[Unit]
Description=CryptoAgent Trading Bot
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=cryptoagent
WorkingDirectory=/home/cryptoagent/cryptoagent
Environment="PYTHONUTF8=1"
Environment="PYTHONUNBUFFERED=1"
ExecStartPre=/bin/sleep 5
ExecStart=/home/cryptoagent/cryptoagent/.venv_312/bin/python -u -m src.agent
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/cryptoagent/cryptoagent/logs/agent.log
StandardError=append:/home/cryptoagent/cryptoagent/logs/agent.log

[Install]
WantedBy=multi-user.target
```

### 7b. Dashboard service

```bash
sudo nano /etc/systemd/system/cryptoagent-dashboard.service
```

```ini
[Unit]
Description=CryptoAgent Streamlit Dashboard
After=network-online.target docker.service cryptoagent.service
Wants=network-online.target

[Service]
Type=simple
User=cryptoagent
WorkingDirectory=/home/cryptoagent/cryptoagent
ExecStart=/home/cryptoagent/cryptoagent/.venv_312/bin/streamlit run \
    src/monitoring/dashboard.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true
Restart=on-failure
RestartSec=15
StandardOutput=append:/home/cryptoagent/cryptoagent/logs/dashboard.log
StandardError=append:/home/cryptoagent/cryptoagent/logs/dashboard.log

[Install]
WantedBy=multi-user.target
```

### 7c. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable cryptoagent cryptoagent-dashboard
sudo systemctl start cryptoagent
sleep 5
sudo systemctl start cryptoagent-dashboard

# Check status
sudo systemctl status cryptoagent
sudo systemctl status cryptoagent-dashboard

# Live logs
journalctl -u cryptoagent -f
# or
tail -f ~/cryptoagent/logs/agent.log
```

---

## 8. Python Environment on VPS

The agent needs a venv with all packages installed. Since the VPS has no GPU, install the CPU-only torch variant (saves 2 GB download, LSTM inference runs fine on CPU):

```bash
cd ~/cryptoagent
python3.12 -m venv .venv_312
source .venv_312/bin/activate

# CPU-only PyTorch (saves ~2 GB vs CUDA build — inference only, no training on VPS)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r requirements.txt
```

---

## 9. Weekly Retraining Workflow (Local → VPS)

Training stays on your local Windows machine (RTX 5070). After each retrain:

### Step 1: Run retrain locally (Windows)
```batch
scripts\retrain_weekly.bat
```

### Step 2: Sync models to VPS
```bash
# Windows Git Bash
rsync -avz --progress \
    /c/Users/ZESTRO/Desktop/cryptoagent/models/ \
    cryptoagent@<VPS_IP>:~/cryptoagent/models/
```

### Step 3: Restart agent to load new models
```bash
ssh cryptoagent@<VPS_IP> "sudo systemctl restart cryptoagent"
```

### Automate with a local batch script

Create `scripts\deploy_models.bat` on Windows:
```batch
@echo off
echo Syncing models to VPS...
rsync -avz --progress ^
    /c/Users/ZESTRO/Desktop/cryptoagent/models/ ^
    cryptoagent@<VPS_IP>:~/cryptoagent/models/

echo Restarting agent...
ssh cryptoagent@<VPS_IP> "sudo systemctl restart cryptoagent"
echo Done.
```

Add to `retrain_weekly.bat` as the final step — after training completes, auto-deploy.

---

## 10. Log Rotation

Prevent logs growing unbounded:

```bash
sudo nano /etc/logrotate.d/cryptoagent
```

```
/home/cryptoagent/cryptoagent/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    copytruncate
}
```

---

## 11. TimescaleDB Backup

```bash
# Daily backup cron (add via crontab -e as cryptoagent user)
0 3 * * * docker exec cryptoagent_timescaledb pg_dump -U agent cryptoagent \
    | gzip > ~/cryptoagent/backups/db_$(date +\%Y\%m\%d).sql.gz

# Keep last 14 days
0 4 * * * find ~/cryptoagent/backups/ -name "*.sql.gz" -mtime +14 -delete
```

```bash
mkdir -p ~/cryptoagent/backups
crontab -e
# Add both lines above
```

---

## 12. Deployment Checklist

```
[ ] VPS provisioned (Hetzner CPX31, Singapore region)
[ ] SSH key auth working, password auth disabled
[ ] UFW firewall active (22/80/443 only)
[ ] Docker + Docker Compose v2 installed
[ ] git clone + .env uploaded
[ ] models/ rsync'd (40 LGBM + 10 LSTM)
[ ] docker compose up -d timescaledb redis  →  both healthy
[ ] Python venv created, CPU torch + requirements installed
[ ] Domain A record pointing to VPS IP
[ ] Nginx configured + SSL cert issued (certbot)
[ ] Basic auth password set for dashboard
[ ] cryptoagent.service enabled + started
[ ] cryptoagent-dashboard.service enabled + started
[ ] https://dashboard.yourdomain.com loads
[ ] Agent log shows single heartbeat every 5s
[ ] Log rotation configured
[ ] DB backup cron active
[ ] Local deploy_models.bat script tested
[ ] Weekly retrain → rsync → restart workflow verified end-to-end
```

---

## 13. Switching from Paper to Live

When paper validation is complete (target: 50+ trades, positive Sharpe):

```bash
# On VPS
nano ~/cryptoagent/config/settings.yaml
# Change:  mode: paper
# To:      mode: live

sudo systemctl restart cryptoagent
tail -f ~/cryptoagent/logs/agent.log
# Confirm: "Agent mode: live" in startup logs
```

> Start live with reduced position sizing: set `max_single_trade_risk_pct: 0.25`
> (half of paper settings) for the first 2 weeks.

---

## 14. Quick Reference

```bash
# Agent status
sudo systemctl status cryptoagent

# Live logs
tail -f ~/cryptoagent/logs/agent.log

# Restart agent
sudo systemctl restart cryptoagent

# Stop everything
sudo systemctl stop cryptoagent cryptoagent-dashboard
docker compose -f ~/cryptoagent/docker-compose.yml stop

# DB shell
docker exec -it cryptoagent_timescaledb psql -U agent cryptoagent

# Sync models from local (run on Windows)
rsync -avz /c/Users/ZESTRO/Desktop/cryptoagent/models/ cryptoagent@<VPS_IP>:~/cryptoagent/models/
ssh cryptoagent@<VPS_IP> "sudo systemctl restart cryptoagent"
```
