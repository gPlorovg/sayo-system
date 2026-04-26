# Distributed deployment (Stage 2)

This is a sketch for the multi-node setup. It reuses every component from
`deploy/single-node/`, plus:

* **An internal docker registry** (`registry:2`, or any external Harbor /
  GHCR / ECR). Set `INTERNAL_DOCKER_REGISTRY=registry.example.com` on the
  master `model-registry` service. The wrapping pipeline will then push
  every freshly built `actor_image` there.
* **One `worker-manager` per worker node**, each with a unique
  `NODE_ID`. Master and workers must agree on the same `RAY_NAMESPACE`
  (default `sayo`) and the same `MasterRouter` actor name.
* **Worker nodes do NOT host gateway / router / registry**, only
  `ray start --address=<head>:6379` and one `worker-manager` container.
  When Router schedules a session, that worker's manager pulls the
  `actor_image_tag` from the internal registry and `docker run`s it.

## Topology

```
+------------- master -------------+      +-------- worker N --------+
|  ray-head (head node)            |      |  ray (worker node)       |
|  gateway                         |      |  worker-manager (node-N) |
|  router                          |      |  TranscriptActor* (spawned by WM)
|  worker-manager (node-master)    |      |                          |
|  model-registry                  |      +--------------------------+
|  registry:2 (internal docker)    |
+----------------------------------+
```

## Bring-up

1. Start master with the single-node compose, but pass
   `INTERNAL_DOCKER_REGISTRY=registry.example.com` to `model-registry` so
   it pushes wrapped images.
2. On each worker node:

   ```bash
   ray start --address=<master>:6379 --resources='{}'
   docker run --rm -d \
     --name sayo-worker-manager-N \
     --network host \
     -e NODE_ID=node-N \
     -e RAY_ADDRESS=<master>:10001 \
     -e RAY_NAMESPACE=sayo \
     -e RAY_HEAD_HOST=<master>:6379 \
     -e REGISTRY_URL=http://<master>:8000 \
     -e SAYO_NETWORK=host \
     -e SAYO_DISTRIBUTED=1 \
     -v /var/run/docker.sock:/var/run/docker.sock \
     sayo-worker-manager:latest
   ```

3. Register a model from the master once:

   ```bash
   ./sayoctl register-model registry.example.com/sayo-model-nemo:1.0.0
   ```

   The wrapping pipeline pushes the new `actor_image` to
   `registry.example.com/sayo-actor/<id>:...`. Worker nodes will pull it
   on first `spawn`.

## Why this is enough

* Weights are baked into `actor_image`, so worker nodes only need
  `docker pull` (no NFS / S3 / object storage).
* Router queries Registry (HTTP) once per model and caches the manifest;
  hot path stays manifest-free.
* The same `Dockerfile.actor`, `transcript_actor`, and Silero VAD wheel
  ship to every worker. A bad rebuild is a `sayoctl unregister` away.

See [`cluster.yaml`](cluster.yaml) for an optional `ray up` based
provisioning.
