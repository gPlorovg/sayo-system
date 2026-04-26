# Single-node deployment

```bash
cd sayo-backend-system

# 1) prepare Silero VAD weights (one-time)
curl -L -o vad-weights/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# 2) build the shared host base image, then bring everything up
docker compose -f deploy/single-node/docker-compose.yml --profile build-only build host-base
docker compose -f deploy/single-node/docker-compose.yml up -d --build

# 3) register one per-model image (the Registry will wrap it into actor_image)
./sayoctl register-model registry.example.com/sayo-model-nemo:1.0.0

# 4) smoke test: stream microphone audio through the gateway
python ../guidline/stand/client.py --host localhost --port 50051 --mic-live
```

Ray Dashboard: http://localhost:8265.
Registry catalog (debug): http://localhost:8000/v1/admin/state.
