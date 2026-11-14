"""Physical AI Studio's remote trainer on a Modal GPU, for Studio's "direct" training target.

    modal deploy cloud/modal_trainer.py        # or: ./biddy cloud-trainer
    BIDDY_GPU=A100-80GB modal deploy cloud/modal_trainer.py   # bigger GPU (e.g. pi0.5)

Runs the published trainer image built from the same Studio commit as the
laptop (TRAINER_TAG), so Studio and trainer speak the same protocol. Studio
uploads the dataset, the trainer trains on the GPU and exports, and Studio
downloads the model, exactly as with its own trainer container.

The trainer has no authentication of its own and Modal web endpoints are
public, so it is served only under a random secret path:
https://<workspace>--biddy-trainer-trainer.modal.run/<secret>. The secret lives in
the Modal secret `biddy-trainer` and in cloud/.trainer_url on this laptop
(gitignored); anything else gets 404. scripts/setup_modal_trainer.py
creates both and registers the URL in Studio.

Costs: the GPU is billed while a container runs. It starts on the first
request, stays up while Studio follows a job, and stops after SCALEDOWN_S idle.
"""
import os

import modal

TRAINER_TAG = "0.3.0-dev-8f132416"  # matches ~/physical-ai-studio at 8f13241
GPU = os.environ.get("BIDDY_GPU", "L40S")  # 48 GB; SmolVLA/ACT fit easily
SCALEDOWN_S = 20 * 60
STORAGE = "/var/lib/physicalai-trainer"

image = (
    modal.Image.from_registry(f"ghcr.io/open-edge-platform/physicalai-trainer-cuda:{TRAINER_TAG}")
    .env({"TRAINER_STORAGE_DIR": STORAGE, "HF_HOME": f"{STORAGE}/hf"})
)
storage = modal.Volume.from_name("biddy-trainer-storage", create_if_missing=True)
app = modal.App("biddy-trainer")


@app.function(
    image=image,
    gpu=GPU,
    volumes={STORAGE: storage},
    secrets=[modal.Secret.from_name("biddy-trainer")],
    timeout=24 * 3600,          # one request (e.g. the progress stream) may last a whole training run
    scaledown_window=SCALEDOWN_S,
    max_containers=1,           # jobs live in the trainer's memory: every request must reach the same one
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def trainer():
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from trainer.main import app as trainer_app

    secret = os.environ["TRAINER_PATH_SECRET"]
    # A mounted app's startup hook does not run on its own: hand the trainer's
    # (it starts the job queue) to the outer app.
    return Starlette(
        routes=[Mount(f"/{secret}", app=trainer_app)],
        lifespan=lambda _: trainer_app.router.lifespan_context(trainer_app),
    )
