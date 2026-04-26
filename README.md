# sayo-backend-system

Distributed STT backend on Ray with a heavy-actor pattern.

## Layout

```
sayo-backend-system/
  proto/                    # gRPC contract (client <-> gateway)
    sayo.proto              # protocol source
    sayo_pb2.py / sayo_pb2_grpc.py   # generated stubs (do not edit)
  sayo_host/                # host services - never imports sayo_image
    common/                 # structlog, generic helpers
    gateway/                # gRPC entry-point
    router/                 # Ray MasterRouter
    worker_manager/         # Ray WorkerManager:<node_id>
    registry/               # FastAPI Model Registry + wrapping pipeline
  sayo_image/               # only baked into actor_image
    model_repository/       # BaseSTTModel / ModelRepository INTERFACE only
    transcript_actor/       # Ray @ray.remote actor + bootstrap
    vad/                    # Silero ONNX wrapper
  vad-weights/              # silero_vad.onnx (pre-baked into actor_image)
  docker/                   # Dockerfile.{actor,gateway,router,worker_manager,registry}
  deploy/
    single-node/            # docker-compose.yml
    distributed/            # cluster.yaml + README.md (Stage 2)
  sayoctl                   # tiny admin CLI (POST /v1/admin/models, list, unregister)
  pyproject.toml
```

`sayo_host` and `sayo_image` are mutually independent packages.
A pre-commit / ruff rule forbids cross-imports between the two.

## Quick start (single-node)

```bash
# build host services (gateway, router, worker-manager, model-registry, ray-head)
docker compose -f deploy/single-node/docker-compose.yml up -d --build

# register one per-model image; the registry wraps it into actor_image
./sayoctl register-model registry.example.com/sayo-model-nemo:1.0.0

# smoke-test from the guidline reference client
python guidline/stand/client.py --host localhost --port 50051 --mic-live
```

See [server.md](../server.md) for the full architectural story and
`docs/server.md`-style report.
