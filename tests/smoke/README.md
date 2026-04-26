# Smoke test

End-to-end check that the single-node compose stack is alive: Registry has at
least one manifest, Gateway accepts a streaming session, Router/Worker Manager
spawn an actor container, and final transcripts come back.

## Prerequisites

1. Single-node compose is up:

   ```bash
   docker compose -f deploy/single-node/docker-compose.yml up -d --build
   ```

2. One per-model image is registered (this also wraps it into actor_image):

   ```bash
   ./sayoctl register-model registry.example.com/sayo-model-nemo:1.0.0
   ./sayoctl list   # confirm the manifest is there
   ```

3. Repo `guidline/` is on the PYTHONPATH (it ships the reference client).

## Run

```bash
# headless 5s mic capture against the live gateway
python guidline/stand/client.py --host localhost --port 50051 --mic
```

You should see at least one `is_final=True` line and a non-empty
`Last transcript` block in the summary.

## Drive from this repo

`tests/smoke/run.py` glues HealthCheck + a short streaming session
together for CI / one-shot validation:

```bash
python tests/smoke/run.py --host localhost --port 50051 --duration 4
```
