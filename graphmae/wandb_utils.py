"""Opt-in Weights & Biases logging. No-op when ``--wandb`` is off.

Importing wandb is deferred to init time so runs without the package
installed are unaffected unless ``--wandb`` is passed.
"""

import os


def _load_env_local():
    """Load KEY=VALUE lines from .env.local into environ (no override, no dependency)."""
    if os.environ.get("WANDB_API_KEY"):
        return
    for candidate in (os.path.join(os.getcwd(), ".env.local"),
                      os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.local")):
        if not os.path.isfile(candidate):
            continue
        with open(candidate) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                if key.strip() and value:
                    os.environ.setdefault(key.strip(), value)
        return


def init_wandb(args, seed, suffix):
    """Start a W&B run if ``args.wandb``; else return None."""
    _load_env_local()
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed; run `pip install wandb` or drop --wandb") from exc
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", None)
    dataset = getattr(args, "dataset", None) or "eval"
    # ponytail: one run per seed keeps traces comparable; group by dataset when the UI needs it.
    return wandb.init(
        project=getattr(args, "wandb_project", "ProvFusion"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None) or f"{dataset}-{suffix}-seed{seed}",
        group=getattr(args, "wandb_group", None),
        config={**vars(args), "seed": seed},
        mode=mode,
    )


def log(run, metrics, step=None):
    if run is not None:
        run.log(metrics, step=step)


def finish(run):
    if run is not None:
        run.finish()
