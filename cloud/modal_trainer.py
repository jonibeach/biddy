"""Physical AI Studio's remote trainer on a Modal GPU, for Studio's "direct" training target.

    ./biddy cloud-trainer                                   # start it and register it in Studio
    BIDDY_GPU=A100-80GB ./biddy cloud-trainer               # another GPU
    modal run --detach cloud/modal_trainer.py               # what that runs

Runs the published trainer image built from the same Studio commit as the
laptop (TRAINER_TAG), so Studio and trainer speak the same protocol: Studio
uploads the dataset, the trainer trains on the GPU and exports, and Studio
downloads the model, exactly as with its own trainer container.

Why a tunnel and not a Modal web endpoint: web endpoints cut every HTTP
request at about 150 s, and Studio uploads the dataset (several GB) in one
request, which then never finishes. A Modal tunnel (modal.forward) is a plain
TLS connection to the container without that limit.

The trainer has no authentication of its own and the tunnel is public, so it
is served only under a random secret path: https://<random>.modal.host/<secret>.
The secret lives in the Modal secret `biddy-trainer` and in cloud/.trainer_url
on this laptop (gitignored); anything else gets 404.

Costs: the GPU is billed while the function runs. It stops by itself after
IDLE_MINUTES with no requests and an idle GPU, or after 24 h.
"""
import os
import time

import modal

TRAINER_TAG = "0.3.0-dev-8f132416"  # matches ~/physical-ai-studio at 8f13241
# H100 (80 GB) trains SmolVLA about twice as fast as an L40S for a similar total
# cost. BIDDY_GPU=none runs on CPU only: a smoke test without GPU billing.
GPU = os.environ.get("BIDDY_GPU", "H100")
GPU = None if GPU.lower() == "none" else GPU
IDLE_MINUTES = 30
PORT = 8001
STORAGE = "/var/lib/physicalai-trainer"

# The trainer image points TMPDIR at its storage folder, which on Modal is a
# network volume that cannot hold Unix sockets. Python's multiprocessing and
# PyTorch's DataLoader workers create their sockets in TMPDIR, so every worker
# died (OSError 95) and training hung at 0% GPU. /tmp in the container works.
image = (
    modal.Image.from_registry(f"ghcr.io/open-edge-platform/physicalai-trainer-cuda:{TRAINER_TAG}")
    .env({"TRAINER_STORAGE_DIR": STORAGE, "HF_HOME": f"{STORAGE}/hf", "TMPDIR": "/tmp"})
)
storage = modal.Volume.from_name("biddy-trainer-storage", create_if_missing=True)
tunnel_info = modal.Dict.from_name("biddy-trainer-tunnel", create_if_missing=True)
app = modal.App("biddy-trainer")


def build_app(secret: str, activity: dict):
    """The trainer under /<secret>, counting requests for the idle check."""
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from trainer.main import app as trainer_app

    inner = Starlette(
        routes=[Mount(f"/{secret}", app=trainer_app)],
        # A mounted app's startup hook does not run on its own: hand over the
        # trainer's (it starts the job queue).
        lifespan=lambda _: trainer_app.router.lifespan_context(trainer_app),
    )

    async def counted(scope, receive, send):
        if scope["type"] != "http":
            return await inner(scope, receive, send)
        activity["open"] += 1
        activity["last"] = time.time()
        try:
            await inner(scope, receive, send)
        finally:
            activity["open"] -= 1
            activity["last"] = time.time()

    return counted


@app.function(
    image=image,
    gpu=GPU,
    # Loading is CPU-bound (image augmentation in the DataLoader workers: ~110 ms
    # of CPU per sample); Modal's default 17-core ceiling left the H100 idle a
    # third of the time. 24 cores carry 16 workers (./biddy train-policy --num-workers 16).
    cpu=24,
    volumes={STORAGE: storage},
    secrets=[modal.Secret.from_name("biddy-trainer")],
    timeout=24 * 3600,
    max_containers=1,  # jobs live in the trainer's memory: one trainer at a time
)
def serve(idle_minutes: int = IDLE_MINUTES):
    import threading

    import uvicorn

    activity = {"open": 0, "last": time.time()}
    asgi = build_app(os.environ["TRAINER_PATH_SECRET"], activity)
    config = uvicorn.Config(asgi, host="0.0.0.0", port=PORT, log_level="info", timeout_keep_alive=120)
    server = uvicorn.Server(config)

    def gpu_busy() -> bool:
        try:
            from trainer.devices import get_training_devices

            return any(getattr(d, "busy", False) for d in get_training_devices())
        except Exception:  # noqa: BLE001 - when unsure, count as busy
            return True

    def idle_watch():
        # A training job keeps the GPU busy even if Studio stops polling (laptop
        # asleep), so only quit when there are no requests AND the GPU is idle.
        while not server.should_exit:
            time.sleep(60)
            idle_s = time.time() - activity["last"]
            if activity["open"] == 0 and idle_s > idle_minutes * 60 and not gpu_busy():
                print(f"idle for {idle_s / 60:.0f} min, stopping")
                server.should_exit = True

    threading.Thread(target=idle_watch, daemon=True).start()
    with modal.forward(PORT) as tunnel:
        tunnel_info["url"] = tunnel.url
        tunnel_info["started"] = time.time()
        print(f"trainer tunnel up: {tunnel.url}")
        try:
            server.run()
        finally:
            tunnel_info.pop("url", None)


@app.local_entrypoint()
def main(idle_minutes: int = IDLE_MINUTES):
    tunnel_info.pop("url", None)
    serve.spawn(idle_minutes)
    print("starting the trainer (pulling the image and starting a GPU can take a few minutes)...")
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        url = tunnel_info.get("url")
        if url:
            print(f"TRAINER_TUNNEL={url}")
            return
        time.sleep(5)
    raise SystemExit("the trainer did not come up within 20 minutes; see `modal app logs biddy-trainer`")
