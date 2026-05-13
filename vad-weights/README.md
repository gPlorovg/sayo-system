# vad-weights

Silero VAD ONNX asset. `Dockerfile.actor` copies `silero_vad.onnx` into every
wrapped `actor_image` at `/opt/silero/silero_vad.onnx`.

## How to populate (one-time, before building registry / actors)

```bash
curl -L -o vad-weights/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

The placeholder file `silero_vad.onnx.placeholder` exists only to keep the
directory in version control. Replace it (or download the real file next to
it) before building images that run `Dockerfile.actor`.
