# sayo-backend-system

> Russian: [README.ru.md](README.ru.md)

Distributed **speech-to-text (STT)** backend for Sayo. It exposes a **gRPC** API to clients, routes streaming sessions through a **Ray** cluster, and runs each acoustic model inside a **Docker** container that joins Ray as a **heavy actor** (one loaded model per container, optional multi-session sharing).

Design goals:

- **Thin host stack**: gateway, router, worker manager, and model registry run in small images without PyTorch or model-specific weights.
- **Self-contained model workers**: after registration, workers pull a single **actor image** per model; no shared NFS/S3 for weights at runtime.
- **Clear package split**: `sayo_host` (orchestration) must not import `sayo_image` (actor-side code); keep the boundary when extending the project.

For a long-form architecture narrative in Russian, see [`../server.md`](../server.md).

---

## Architecture (overview)

| Component | Role |
|-----------|------|
| **Gateway** (`sayo_host/gateway`) | gRPC entry point: `HealthCheck`, `StreamingRecognize`. Talks to the registry over HTTP and to Ray for routing and actor calls. |
| **MasterRouter** (`sayo_host/router`) | Ray detached named actor: session placement, actor reuse, VRAM-aware placement, LRU eviction of idle actors. Caches model manifests from the registry. |
| **WorkerManager** (`sayo_host/worker_manager`) | One Ray actor per physical node: `docker run` / stop for actor containers; heartbeats GPU and actor list to the router. |
| **Model registry** (`sayo_host/registry`) | FastAPI service: catalog of manifests, admin registration that **wraps** a per-model image into an **actor image**, optional push to an internal image registry. |
| **TranscriptActor** (`sayo_image/…`) | Runs **inside** the actor image: loads the model, optional Silero VAD (ONNX), streams transcripts back to the gateway. |

**Request path (simplified):** client → gateway → `MasterRouter.acquire_session` → (optional) `WorkerManager.spawn` → container bootstraps Ray worker + `TranscriptActor` → gateway `open_session` / `feed` / `next_result` / `close_session` → `release_session` on the router.

---

## Repository layout

```
sayo-backend-system/
  proto/                      # gRPC contract (client ↔ gateway)
    sayo.proto                # source of truth (edit, then regenerate stubs)
    sayo_pb2.py               # generated — do not hand-edit
    sayo_pb2_grpc.py          # generated — do not hand-edit
  sayo_host/                  # host services — never import sayo_image
    common/                   # logging, shared env helpers
    gateway/                  # gRPC server
    router/                   # MasterRouter bootstrap + actor
    worker_manager/           # WorkerManager bootstrap + actor
    registry/                 # FastAPI app, catalog, wrapper pipeline, admin UI
  sayo_image/                 # baked only into actor_image
    model_repository/         # model.yaml → adapter (BaseSTTModel contract)
    transcript_actor/         # Ray actor, audio helpers, bootstrap entrypoint
    vad/                      # Silero ONNX wrapper
  vad-weights/                # silero_vad.onnx (copied into actor_image)
  docker/                     # Dockerfiles for all runnable images
  deploy/
    single-node/              # docker-compose.yml (default dev / demo topology)
    distributed/              # multi-node notes + cluster.yaml sketch
  tests/smoke/                # end-to-end smoke script
  sayoctl                     # CLI for registry admin HTTP API
  pyproject.toml
  requirements.txt            # locked export (uv); optional for pip-based installs
```

---

## Prerequisites

- **Docker** and **Docker Compose** v2 for the recommended path.
- **Python ≥ 3.12** if you run `sayoctl`, smoke tests, or local tooling outside containers.
- **NVIDIA Container Toolkit** if you use GPU-backed models and set `SAYO_NUM_GPUS` (see compose).
- Per-model images must be built so the registry can read **`MODEL_NAME`** from image env and find `model.yaml` under `/app/models/<MODEL_NAME>/` inside the image (see wrapping pipeline in `sayo_host/registry/wrapper.py`).

**Python / Ray version alignment:** the Ray head and any actor container should use **compatible Ray and Python builds**. Mismatched Ray or Python versions between the head node and a joining worker commonly cause connection or serialization failures.

---

## Quick start (single node)

From the `sayo-backend-system` directory:

```bash
# Build and start: Ray head, registry, router, worker-manager, gateway
docker compose -f deploy/single-node/docker-compose.yml up -d --build
```

Default **published ports** (host → container):

| Port | Service |
|------|---------|
| 50051 | gRPC gateway |
| 8000 | Model registry (HTTP) |
| 8081 | Router admin HTTP (cluster snapshot) |
| 8082 | Worker-manager admin HTTP |
| 6379 | Ray GCS / internal client port |
| 10001 | Ray Client (`ray://…`) |
| 8265 | Ray dashboard |

Register a **per-model** image; the registry builds/tags an **actor image** and writes a manifest:

```bash
./sayoctl register-model registry.example.com/sayo-model-nemo:1.0.0
# or a local tag if the image already exists on the daemon:
./sayoctl register-model my-nemo:1.0.0 --local
```

Smoke test (expects at least one registered model):

```bash
python tests/smoke/run.py --host localhost --port 50051 --duration 4
```

More detail: [`tests/smoke/README.md`](tests/smoke/README.md).

### Building `actor-runtime` (optional but important)

Actor registration runs `docker build` using `docker/Dockerfile.actor`, which expects a pre-built **`sayo-actor-runtime:latest`** image on the same Docker daemon (wheels + generated proto stubs). If dependencies or `proto/sayo.proto` change, rebuild:

```bash
docker compose -f deploy/single-node/docker-compose.yml --profile build-only build actor-runtime
```

---

## Client API (gRPC)

Defined in **`proto/sayo.proto`** (generate stubs after edits).

- **`HealthCheck`** — empty request; response includes `ready`, `message`, and `repeated ModelDescriptor` built from registry manifests (`GET /v1/models` on the gateway side).
- **`StreamingRecognize`** — **bidirectional stream**:
  - First client message **must** be `StreamingConfig` (model id, language, interim flag, audio format, VAD settings).
  - Following messages are raw **mono PCM** chunks (`bytes`) matching the config.
  - Server streams `StreamingRecognizeResponse` (`transcript`, `is_final`, `confidence`, string `metadata` map). Early responses may carry lifecycle hints in `metadata["connection_status"]` (see comments in the proto file).

Regenerate Python stubs after changing the proto (dev dependency `grpcio-tools`, same major as `grpcio`):

```bash
cd sayo-backend-system
uv sync --group dev
uv run python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. proto/sayo.proto
```

This overwrites `proto/sayo_pb2.py` and `proto/sayo_pb2_grpc.py` next to `sayo.proto`, matching `from proto import sayo_pb2` in the codebase.

---

## Model registry (HTTP)

Public read API:

- `GET /v1/health` — liveness and model count.
- `GET /v1/models` — list manifests (JSON).
- `GET /v1/models/{model_id}` — single manifest.

Admin / operator API:

- `POST /v1/admin/models` — register (`image_ref`, optional `force`, `local_only`).
- `POST /v1/admin/models/register-stream` — same with **NDJSON** progress lines.
- `DELETE /v1/admin/models/{model_id}` — remove from catalog (optional image removal via query flags where implemented).
- `GET /v1/admin/state` — debug snapshot.
- `GET /admin` — small HTML UI (if packaged).

**Manifest** fields (conceptually): `model_id`, `actor_image_tag`, `model_dir`, audio/runtime fields, `min_vram_gb`, `max_concurrent_sessions`, `vad` block, etc. Gateway and router consume manifests **only via HTTP**, not by reading YAML from disk.

**Distributed mode:** set `INTERNAL_DOCKER_REGISTRY` on the registry service so freshly built actor images are **pushed** to your private registry; worker nodes pull on first spawn. See [`deploy/distributed/README.md`](deploy/distributed/README.md).

---

## `sayoctl` (CLI)

Installed as a console script from `pyproject.toml` (`pip install -e .` / `uv sync`). Targets `SAYO_REGISTRY_URL` (default `http://localhost:8000`).

Typical commands:

```bash
sayoctl register-model <image_ref> [--local] [--force]
sayoctl list
sayoctl unregister <model_id> [--remove-image]
sayoctl state
```

---

## Environment variables (reference)

Values below are typical **defaults or compose defaults**; override in your environment as needed.

| Variable | Used by | Meaning |
|----------|---------|---------|
| `RAY_ADDRESS` | gateway, router, worker-manager | Ray Client address, e.g. `ray://ray-head:10001`. |
| `RAY_NAMESPACE` | all Ray clients | Logical namespace for named actors (default `sayo`). |
| `REGISTRY_URL` | gateway, router, worker-manager | Base URL of the model registry HTTP API. |
| `NODE_ID` | worker-manager | Unique node id (`WorkerManager:<NODE_ID>`). |
| `RAY_HEAD_HOST` | worker-manager, actor cmdline | Ray address passed into actor containers, e.g. `ray-head:6379`. |
| `SAYO_NETWORK` | worker-manager | Docker network name for spawned actor containers (compose may set `sayo_sayo_net`). |
| `SAYO_DISTRIBUTED` | worker-manager | `1` to `docker pull` actor images before run. |
| `GATEWAY_PORT` | gateway | gRPC listen port (default `50051`). |
| `REGISTRY_ROOT` | registry | Manifest catalog directory (default `/manifests`). |
| `INTERNAL_DOCKER_REGISTRY` | registry | If set, push built actor images to this registry. |
| `SAYO_REPO_ROOT` | registry build | Filesystem root containing `sayo_image`, `docker`, `proto`. |
| `SAYO_ACTOR_RUNTIME_IMAGE` | registry | Override actor-runtime image tag used during wrap build. |
| `SAYO_ACTOR_REGISTER_TIMEOUT` | worker-manager | Seconds to wait for actor Ray registration after `docker run`. |
| `SAYO_ACTOR_SHM_SIZE` | worker-manager | Docker `shm_size` for actor containers (default `2g`). |
| `SAYO_DISABLE_VAD` | gateway (and actor) | When truthy, ignore client VAD / disable VAD paths for debugging. |
| `SAYO_REPLACE_MASTER_ROUTER` | router bootstrap | If truthy, kill existing `MasterRouter` before recreating (dangerous in prod). |
| `ADMIN_PORT` | router, worker-manager | HTTP port for optional admin snapshot server. |

---

## Multi-node (distributed)

Stage-2 topology: one **master** host runs Ray head, gateway, router, registry, and a worker-manager; additional **worker** hosts run Ray workers plus their own worker-manager with a distinct `NODE_ID`. Worker hosts do not need the gateway or registry containers if they can reach the master’s Ray address and image registry.

Step-by-step notes: [`deploy/distributed/README.md`](deploy/distributed/README.md).

---

## Local development (without full compose)

```bash
uv sync
# or: pip install -e .
```

Run linters / formatters (see `pyproject.toml` and `.pre-commit-config.yaml`):

```bash
pre-commit run --all-files
```

When adding features, keep **`sayo_host` independent of `sayo_image`** so host images stay lightweight and actor images remain the single place for model adapters and heavy dependencies.

---

## Troubleshooting (short)

- **Gateway `ready=false` in HealthCheck** — registry unreachable from gateway; check `REGISTRY_URL` and network.
- **Session stalls or no actor** — inspect Ray dashboard (`8265`), router admin (`8081`), and worker-manager admin (`8082`); verify manifests list `actor_image_tag` and that the worker daemon can pull/run that image.
- **Actor fails to register in Ray** — often image/Python/Ray mismatch, insufficient `shm_size`, or GPU scheduling (`device` / `num-gpus`) inconsistent with the node.
- **Registration build fails** — ensure `sayo-actor-runtime:latest` exists (`build-only` compose profile) and Docker socket is mounted where the registry runs.

---

## Versions and metadata

Project metadata and dependencies are declared in **`pyproject.toml`**. Ray is pinned (`ray[default]==2.39.0`); upgrade deliberately and rebuild all images that embed Ray.
