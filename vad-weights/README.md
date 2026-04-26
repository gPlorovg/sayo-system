# vad-weights

Pre-baked Silero VAD weights folder. The Model Registry's `Dockerfile.actor`
copies `silero_vad.onnx` from this directory into every wrapped `actor_image`
under `/opt/silero/silero_vad.onnx`.

## How to populate (one-time, before first `register-model`)

```bash
# ~1.7 MB; pinned commit guarantees reproducibility
curl -L -o vad-weights/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

The placeholder file `silero_vad.onnx.placeholder` exists only to keep the
directory in version control. Replace it (or download the real weights next
to it) before building any actor_image.
