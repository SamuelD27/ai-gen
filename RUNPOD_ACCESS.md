# RunPod Access Guide

## How to Access Your ai-gen GUI on RunPod

After running `bash runpod_quickstart.sh`, your services are running on:
- **Frontend (Vue GUI)**: Port 5173
- **Backend (FastAPI)**: Port 8000

### Method 1: HTTP Service (Recommended)

1. In your RunPod pod dashboard, click the **"Connect"** button
2. Look for **"HTTP Service [Port 8000]"** or **"HTTP Service [Port 5173]"**
3. Click on the HTTP link - it will look like:
   - `https://YOUR-POD-ID-8000.proxy.runpod.net` (Backend)
   - Or RunPod may auto-detect port 5173 for the frontend

### Method 2: Direct TCP Port

If you see "Direct TCP Ports" with something like `149.36.1.226:33365`:

This is an **SSH port mapping** (port 22), not for the GUI. For web access, use Method 1.

### Method 3: SSH Port Forwarding (Advanced)

If HTTP Service doesn't work, create an SSH tunnel:

```bash
# On your local machine
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 root@149.36.1.226 -p 33365
```

Then access on your local machine:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000/docs

### Method 4: Expose Custom Ports

Edit your RunPod template to expose ports:
- Add **5173** for frontend
- Add **8000** for backend

Then restart your pod.

### Troubleshooting

**Q: The HTTP Service link shows "Connection refused"**
- Make sure the services are running: `ps aux | grep uvicorn`
- Check logs: `cd /workspace/ai-gen/charforge-gui && tail -f nohup.out`

**Q: I see 404 or blank page**
- The backend (port 8000) serves the API, not the GUI
- Try accessing the frontend (port 5173) instead
- Use: `http://YOUR-POD-ID-5173.proxy.runpod.net`

**Q: How do I know what my proxy URL is?**
- RunPod automatically creates it in the format: `https://YOUR-POD-ID-PORT.proxy.runpod.net`
- Check the "Connect" button in your pod dashboard for the exact URL

### Quick Check

Run this in your RunPod terminal to verify services are running:

```bash
curl http://localhost:8000/docs
curl http://localhost:5173
```

Both should return HTML content.
