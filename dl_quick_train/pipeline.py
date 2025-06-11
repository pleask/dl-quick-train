import json
from functools import partial
import multiprocessing as mp
import os
import queue

import torch
from datasets import load_dataset
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from nnsight import LanguageModel
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb


def new_wandb_process(
    config,
    log_queue,
    entity,
    project,
    run_id_queue=None,
    index=None,
):
    run = wandb.init(
        entity=entity, project=project, config=config, name=config["wandb_name"]
    )
    if run_id_queue is not None:
        run_id_queue.put((index, run.id))
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            if isinstance(log, dict):
                step = log.pop("step", None)
                wandb.log(log, step=step)
            elif isinstance(log, tuple) and log[0] == "artifact":
                artifact_path = log[1]
                artifact_name = os.path.basename(artifact_path)
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(artifact_path)
                run.log_artifact(artifact)
        except queue.Empty:
            continue
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: torch.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list = [],
    verbose: bool = False,
):
    with torch.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone().to(trainer.device)
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # Number of unique SAE features active in the batch
                # f can be shape (batch, seq, features) or (batch, features)
                active_mask = f != 0
                if active_mask.ndim > 2:
                    active_mask = active_mask.reshape(-1, active_mask.shape[-1])
                unique_active = active_mask.any(dim=0).sum().item()
                # fraction of variance explained
                total_variance = torch.var(act, dim=0).sum()
                residual_variance = torch.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # Number of unique SAE features active in the batch
                active_mask = f != 0
                if active_mask.ndim > 2:
                    active_mask = active_mask.reshape(-1, active_mask.shape[-1])
                unique_active = active_mask.any(dim=0).sum().item()

            if verbose:
                print(
                    f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}"
                )

            # log parameters from training
            log.update(
                {
                    f"{k}": v.cpu().item() if isinstance(v, torch.Tensor) else v
                    for k, v in losslog.items()
                }
            )
            log[f"l0"] = l0
            log["unique_active_features"] = unique_active
            log["step"] = step
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def collate(batch, tok, seq_len):
    # tok comes from parent, safe to pickle; do the heavy work here
    return tok(
        [ex["text"] for ex in batch],
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )["input_ids"]


def run_pipeline(
    trainer_configs,
    *,
    device="cuda:0",
    model_name="EleutherAI/pythia-70m-deduped",
    submodule=None,
    dataset_name="HuggingFaceFW/fineweb",
    steps=100_000,
    batch_size=128,
    seq_len=32,
    run_cfg={},
    use_wandb=False,
    use_transformer_lens=False,
    stop_at_layer=None,
    wandb_entity=None,
    wandb_project=None,
    save_dir=None,
    log_steps=100,
    verbose=False,
    save_steps=None,
    **kwargs,
):
    mp.set_start_method("spawn", force=True)
    if use_transformer_lens:
        model = HookedTransformer.from_pretrained(model_name, device=device)
    else:
        model = LanguageModel(model_name, device_map=device)
    dataset = load_dataset(dataset_name, streaming=True, split="train")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.backend_tokenizer.enable_truncation(max_length=seq_len)
    tok.backend_tokenizer.enable_padding(
        length=seq_len, pad_id=tok.pad_token_id, pad_token=tok.pad_token
    )

    if not use_transformer_lens:
        submodule_ref = eval(f"model.{submodule}")

    trainers = []
    for cfg in trainer_configs:
        cls = cfg.pop("trainer")
        trainer = cls(**cfg)
        trainer.ae = trainer.ae.to(device)
        trainers.append(trainer)


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=8,
        persistent_workers=True,
        collate_fn=partial(collate, tok=tok, seq_len=seq_len),
    )
    loader = iter(loader)

    log_queues = []
    wandb_processes = []
    run_id_queue = mp.Queue()
    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue(32)
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            wandb_config = {
                k: v.cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in wandb_config.items()
            }
            p = mp.Process(
                target=new_wandb_process,
                args=(
                    wandb_config,
                    log_queue,
                    wandb_entity,
                    wandb_project,
                    run_id_queue,
                    i,
                ),
            )
            p.start()
            wandb_processes.append(p)

    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, trainer_dir in zip(trainers, save_dirs):
            os.makedirs(trainer_dir, exist_ok=True)
            config = {"trainer": trainer.config}
            with open(os.path.join(trainer_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    stream = torch.cuda.Stream()
    for step in tqdm(range(steps), desc="Training"):
        batch = next(loader)
        with torch.cuda.stream(stream):
            with torch.no_grad():
                if use_transformer_lens:
                    _, cache = model.run_with_cache(
                        batch.to(device),
                        names_filter=[submodule],
                        stop_at_layer=stop_at_layer,
                    )
                    act = cache[submodule]
                else:
                    with model.trace(
                        batch.to(device), invoker_args={"max_length": seq_len}
                    ):
                        h = submodule_ref.output.save()
                        submodule_ref.output.stop()
                    act = h.value[0]

        for tnr in trainers:
            if (use_wandb or verbose) and step % log_steps == 0:
                log_stats(
                    trainers,
                    step,
                    act,
                    False,
                    False,
                    log_queues=log_queues,
                    verbose=verbose,
                )

            if save_steps is not None and step in save_steps:
                for idx, (trainer_dir, trainer) in enumerate(zip(save_dirs, trainers)):
                    if trainer_dir is None:
                        continue
                    if not os.path.exists(os.path.join(trainer_dir, "checkpoints")):
                        os.mkdir(os.path.join(trainer_dir, "checkpoints"))
                    checkpoint = {
                        k: v.cpu() for k, v in trainer.ae.state_dict().items()
                    }
                    path = os.path.join(trainer_dir, "checkpoints", f"ae_{step}.pt")
                    torch.save(
                        checkpoint,
                        path,
                    )
                    if use_wandb:
                        log_queues[idx].put(("artifact", path))

            tnr.update(step, act)

    if use_wandb:
        for q in log_queues:
            q.put("DONE")
        for p in wandb_processes:
            p.join()
        run_ids = [None] * len(trainer_configs)
        for _ in range(len(trainer_configs)):
            idx, r_id = run_id_queue.get()
            run_ids[idx] = r_id
    else:
        run_ids = []
    return run_ids
