# Security & Deployment Guide

## 🔐 Current Security Posture

DriftDetector v2 is a **monitoring dashboard**, not an auth-gated service. 

**What's Protected:**
- ✓ CDN integrity (Chart.js pinned + signed)
- ✓ Threshold validation (env config verified at startup)
- ✓ Error handling (no stack traces in API responses)
- ✓ Thread-safe singleton (dependency injection)

**What's NOT Protected (by design):**
- ✗ No API authentication (endpoints are public)
- ✗ No CORS restrictions (default: allow all)
- ✗ No rate limiting (add via nginx/proxy if needed)

---

## 🚀 Deployment Modes

### 1. Local Development (Default)
```bash
UI_HOST=127.0.0.1
UI_PORT=8000
CORS_ORIGINS=*
```
Only accessible on localhost. Safe for dev/testing.

### 2. Internal Network (Team/Company)
```bash
UI_HOST=0.0.0.0
UI_PORT=8000
CORS_ORIGINS=https://internal.company.com
```
Accessible within your network. Restrict CORS to your domain.

### 3. Production (Public)
**DO NOT expose without these steps:**

1. **Add authentication layer (nginx/Cloudflare/AWS ALB):**
   ```nginx
   location /api/ {
       auth_basic "restricted";
       auth_basic_user_file /etc/nginx/.htpasswd;
       proxy_pass http://localhost:8000;
   }
   ```

2. **Restrict CORS:**
   ```bash
   CORS_ORIGINS=https://app.yourdomain.com,https://monitor.yourdomain.com
   ```

3. **Use HTTPS only:**
   ```bash
   # Your reverse proxy handles this
   # DriftDetector runs on HTTP internally
   ```

4. **Rate limit (nginx):**
   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
   limit_req zone=api burst=20;
   ```

---

## 📋 What Each Endpoint Exposes

| Endpoint | Visibility | Risk | Mitigation |
|----------|------------|------|-----------|
| `GET /` | HTML dashboard | Low | Served as static file |
| `GET /api/health` | Detector status | Low | Basic info only |
| `GET /api/config` | Threshold values | Low | Public config |
| `POST /api/drift` | Measurement endpoint | **HIGH** | Input validation required |
| `GET /api/chain` | Full history | **HIGH** | Exposes all drift reports |

**High-Risk Endpoints:**
- `POST /api/drift` accepts arbitrary text input → sanitize if exposing
- `GET /api/chain` returns full drift history → consider pagination/filtering

---

## 🛡️ Recommended Production Setup

```
Client (HTTPS)
    ↓
[Nginx/CloudFlare/AWS ALB]
    - Authentication (OAuth/SSO/Basic Auth)
    - Rate limiting
    - CORS validation
    - HTTPS termination
    ↓
DriftDetector v2 (HTTP, internal only)
    - No auth needed (proxy handles it)
    - Localhost binding (127.0.0.1)
    - Internal API only
```

---

## 🔑 Environment Variables (Security)

**Sensitive values NOT stored in code:**
```bash
DRIFT_THRESHOLD=0.6          # Config, not secret
SIGNAL_THRESHOLD=0.7         # Config, not secret
CORS_ORIGINS=https://...     # Your domain
UI_HOST=127.0.0.1           # Binding (default: localhost only)
```

**No API keys needed** for drift detection (local analysis only).

---

## ⚠️ Known Limitations

1. **No built-in authentication** — Add at proxy layer
2. **No rate limiting** — Add via reverse proxy (nginx/Cloudflare)
3. **No input validation** on `/api/drift` text fields — Sanitize if public
4. **No encryption in transit** — Use HTTPS reverse proxy
5. **No audit logging** — Detector doesn't log who called what

---

## ✅ Security Checklist

- [ ] Running on `127.0.0.1` in dev (not `0.0.0.0`)
- [ ] CORS restricted to specific domains in prod
- [ ] Reverse proxy (nginx/ALB) handles authentication
- [ ] HTTPS enabled on reverse proxy
- [ ] Rate limiting configured
- [ ] Chart.js integrity verified (SRI enabled)
- [ ] Environment config validated at startup
- [ ] Error messages don't leak stack traces

---

**For questions:** See `README.md` or open an issue.

---

## 📮 Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security reports.**

We take security seriously. If you believe you have found a security
vulnerability in DriftDetector, report it privately so we can fix it
before it is publicly disclosed.

### How to report

Choose one of these channels:

1. **GitHub Security Advisories (preferred):**
   Open a private advisory at
   <https://github.com/MrPredic/drift-detector/security/advisories/new>

   This is the fastest path — GitHub notifies the maintainer immediately
   and keeps the report private until a fix is released.

Please include:
- A description of the issue and its impact
- Steps to reproduce (proof-of-concept where possible)
- The affected version(s) (see `VERSION` / `drift_detector/__version__`)
- Your name/handle if you'd like to be credited

### What to expect

| Step | Target |
|------|--------|
| Acknowledgement of receipt | within 72 hours |
| Initial severity assessment | within 7 days |
| Fix + coordinated disclosure | typically within 30 days for High/Critical |

We will keep you updated throughout, credit you in the release notes
(unless you prefer to stay anonymous), and never take legal action
against researchers acting in good faith.

### Supported versions

Only the latest minor release of the 2.x line receives security fixes.
See `CHANGELOG.md` for the current release.

### Scope

In scope: the `drift-detector` Python package, its optional FastAPI
dashboard, and the shipped HTML frontend.

Out of scope: third-party LLM providers, LangChain/CrewAI internals,
CDN-hosted JavaScript libraries (report those to their own maintainers).

