"""Entrypoint of actor_image. Joins the Ray cluster as a worker and pins
TranscriptActor to *this* container via a UUID custom resource.

Steps:
  1. `ray start --address ... --resources={"<slot>":1}` (subprocess).
  2. `ray.init(address="auto")`.
  3. Create detached named `TranscriptActor` with `resources={"<slot>":1}`
     so Ray schedules the actor on this exact container.
  4. Block forever; on SIGTERM gracefully `ray.kill(actor)` + `ray stop`.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time

import ray
import structlog

from sayo_image.transcript_actor.log_config import configure_actor_process_logging

configure_actor_process_logging()
logger = structlog.get_logger("transcript_actor.bootstrap")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sayo TranscriptActor bootstrap")
    p.add_argument("--ray-address", required=True, help="ray-head:6379")
    p.add_argument("--actor-name", required=True)
    p.add_argument("--namespace", default="sayo")
    p.add_argument("--model-name", required=True, help="Resolves /app/models/<name>")
    p.add_argument(
        "--model-dir",
        default=None,
        help="Override; default /app/models/<model-name>",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--slot-resource", required=True, help="UUID custom resource name")
    p.add_argument("--max-concurrent-sessions", type=int, default=4)
    return p.parse_args()


def _start_ray_worker(ray_address: str, slot_resource: str, device: str) -> None:
    resources = json.dumps({slot_resource: 1})
    cmd = ["ray", "start", "--address", ray_address, "--resources", resources]
    # TranscriptActor requests num_gpus=1 on CUDA; the worker must advertise a GPU
    # or Ray will never schedule the actor on this node.
    if device.startswith("cuda"):
        cmd.extend(["--num-gpus", "1"])
    logger.info("starting ray worker", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)


def _stop_ray_worker() -> None:
    try:
        subprocess.run(["ray", "stop", "--force"], check=False, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("ray stop timed out")


def main() -> None:
    args = _parse_args()
    model_dir = args.model_dir or f"/app/models/{args.model_name}"

    _start_ray_worker(args.ray_address, args.slot_resource, args.device)
    # Show worker/actor logs in container stdout to make debugging model init easier.
    # (Ray Client streaming isn't used here; this is inside the actor container.)
    ray.init(address="auto", namespace=args.namespace, log_to_driver=True)

    from sayo_image.transcript_actor.actor import TranscriptActor

    num_gpus = 1.0 if args.device.startswith("cuda") else 0.0
    actor = TranscriptActor.options(
        name=args.actor_name,
        namespace=args.namespace,
        lifetime="detached",
        resources={args.slot_resource: 1},
        num_gpus=num_gpus,
        max_concurrency=max(8, args.max_concurrent_sessions * 2),
    ).remote(
        model_dir=model_dir,
        device=args.device,
        max_concurrent_sessions=args.max_concurrent_sessions,
    )

    logger.info(
        "TranscriptActor created",
        actor_name=args.actor_name,
        model_dir=model_dir,
        device=args.device,
        slot_resource=args.slot_resource,
    )

    stop = False

    def _shutdown(signum, _frame):
        nonlocal stop
        logger.info("signal received", signum=signum)
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while not stop:
            time.sleep(1.0)
    finally:
        try:
            ray.get(actor.unload.remote(), timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("actor unload failed", error=str(exc))
        try:
            ray.kill(actor, no_restart=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ray.kill failed", error=str(exc))
        _stop_ray_worker()
        logger.info("bootstrap exited cleanly")


if __name__ == "__main__":
    sys.exit(main())
