## Oracle VM Deployment

This is the recommended full-deployment path for Hydrovision when you want:

- the full Flask app online
- the full dataset available locally on the server
- lower monthly cost than managed app hosting with persistent disk

This guide assumes:

- Ubuntu 22.04 on an Oracle Always Free VM
- app code at `/opt/hydrovision/app`
- data at `/opt/hydrovision/data`
- Python virtualenv at `/opt/hydrovision/venv`

### 1. Create the VM

Use an Oracle Always Free VM:

- shape: Ampere A1 or small AMD Always Free shape
- image: Ubuntu 22.04
- open ingress ports:
  - `22`
  - `80`
  - `443`

### 2. SSH into the VM

```bash
ssh -i /path/to/private_key ubuntu@<VM_PUBLIC_IP>
```

### 3. Install base packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git unzip
```

### 4. Clone the repo

```bash
sudo mkdir -p /opt/hydrovision
sudo chown -R $USER:$USER /opt/hydrovision
git clone https://github.com/hungdn210/hydrovision.git /opt/hydrovision/app
cd /opt/hydrovision/app
```

### 5. Create the virtual environment

```bash
python3 -m venv /opt/hydrovision/venv
source /opt/hydrovision/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Copy environment file

```bash
cp deploy/oracle/hydrovision.env.example /opt/hydrovision/hydrovision.env
```

Edit it:

```bash
nano /opt/hydrovision/hydrovision.env
```

Set at least:

- `HYDROVISION_DATA_DIR=/opt/hydrovision/data`
- `GEMINI_API_KEY=...` if you want AI analysis

If you want the VM to sync the dataset from Cloudflare R2, also set:

- `R2_BUCKET`
- `R2_ENDPOINT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

### 7. Populate the data directory

Recommended destination:

```bash
mkdir -p /opt/hydrovision/data
```

You have two options:

#### Option A: sync from R2

```bash
source /opt/hydrovision/venv/bin/activate
set -a
source /opt/hydrovision/hydrovision.env
set +a
python /opt/hydrovision/app/scripts/sync_r2_data.py
```

#### Option B: copy data directly to the VM

If you already have the data locally and want a one-time transfer:

```bash
rsync -avz /local/path/to/data/ ubuntu@<VM_PUBLIC_IP>:/opt/hydrovision/data/
```

### 8. Install the systemd service

```bash
sudo cp deploy/oracle/hydrovision.service /etc/systemd/system/hydrovision.service
sudo systemctl daemon-reload
sudo systemctl enable hydrovision
sudo systemctl start hydrovision
sudo systemctl status hydrovision
```

### 9. Install the nginx site

```bash
sudo cp deploy/oracle/hydrovision.nginx /etc/nginx/sites-available/hydrovision
sudo ln -sf /etc/nginx/sites-available/hydrovision /etc/nginx/sites-enabled/hydrovision
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 10. Update the server IP in nginx

Edit:

```bash
sudo nano /etc/nginx/sites-available/hydrovision
```

Replace:

```text
server_name _;
```

with your domain if you have one:

```text
server_name your-domain.example;
```

### 11. Logs

App logs:

```bash
sudo journalctl -u hydrovision -f
```

Nginx logs:

```bash
sudo tail -f /var/log/nginx/error.log
```

### 12. Redeploy after code changes

```bash
cd /opt/hydrovision/app
git pull
source /opt/hydrovision/venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart hydrovision
```

### 13. Optional HTTPS

After nginx is working, install Certbot:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.example
```
