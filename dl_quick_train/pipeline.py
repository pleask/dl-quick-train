import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
import faulthandler
from typing import Optional

import torch
import torch as t  # Hate this - leftover from dl
from datasets import load_dataset
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb

MAX_ERRORS = 5


def input_fetcher(
    q,
    model_name,
    dataset_name,
    steps,
    batch_size,
    max_length,
    device,
    max_errors: int = MAX_ERRORS,
    error_queue: mp.Queue = None,
):
    faulthandler.enable()
    dataset = load_dataset(dataset_name, streaming=True, split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=8,
        persistent_workers=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.backend_tokenizer.enable_truncation(max_length=max_length)
    tok.backend_tokenizer.enable_padding(
        length=max_length, pad_id=tok.pad_token_id, pad_token=tok.pad_token
    )
    errors = 0
    with tqdm(total=steps, desc="Training...") as pbar:
        iterator = iter(loader)
        i = 0
        while i < steps:
            try:
                batch = next(iterator)
                inputs = tok(
                    batch["text"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                )
                inputs = {
                    k: v.to("cpu", non_blocking=True).share_memory_()
                    for k, v in inputs.items()
                }
                q.put(inputs)
                pbar.update(1)
                i += 1
            except Exception as e:
                errors += 1
                tb = traceback.format_exc()
                if error_queue is not None:
                    error_queue.put(("input_fetcher", tb))
                traceback.print_exc()
                if errors >= max_errors:
                    print("input_fetcher: too many errors, exiting")
                    if error_queue is not None:
                        error_queue.put(("input_fetcher", "too many errors"))
                    sys.exit(1)
                else:
                    print("input_fetcher: error encountered, skipping batch")
                    continue
    q.put("DONE")


def activation_fetcher(
    in_q,
    out_q,
    model_name,
    submodule,
    ctx_len,
    device,
    cache_dir=None,
    max_errors: int = MAX_ERRORS,
    error_queue: mp.Queue = None,
):
    faulthandler.enable()
    model = LanguageModel(model_name, device_map=device)
    model.eval()
    submodule = eval(f"model.{submodule}")  # caller decides string

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    step_idx = 0

    errors = 0
    while True:
        try:
            inputs = in_q.get(timeout=1)
        except queue.Empty:
            continue
        except (EOFError, ConnectionResetError, BrokenPipeError):
            msg = (
                "activation_fetcher: lost connection to input_fetcher (possible crash)"
            )
            if error_queue is not None:
                error_queue.put(("activation_fetcher", msg))
            print(msg, file=sys.stderr)
            return
        if inputs == "DONE":
            break
        try:
            with torch.inference_mode():
                with model.trace(inputs, invoker_args={"max_length": ctx_len}):
                    h = submodule.output.save()
                    submodule.output.stop()
                act = h.value.to("cpu", non_blocking=True)
                out_q.put(act.share_memory_())

            if cache_dir is not None:
                torch.save(act, os.path.join(cache_dir, f"{step_idx}.pt"))
                step_idx += 1
        except (EOFError, ConnectionResetError, BrokenPipeError):
            msg = "activation_fetcher: lost connection to train (possible crash)"
            if error_queue is not None:
                error_queue.put(("activation_fetcher", msg))
            print(msg, file=sys.stderr)
            return
        except Exception:
            errors += 1
            tb = traceback.format_exc()
            if error_queue is not None:
                error_queue.put(("activation_fetcher", tb))
            traceback.print_exc()
            if errors >= max_errors:
                print("activation_fetcher: too many errors, exiting")
                if error_queue is not None:
                    error_queue.put(("activation_fetcher", "too many errors"))
                sys.exit(1)
            else:
                print("activation_fetcher: error encountered, skipping batch")
                continue
    out_q.put("DONE")


def activation_loader(out_q, cache_dir, steps):
    """Load cached activations from disk and feed them into the pipeline."""
    faulthandler.enable()
    for i in range(steps):
        act = torch.load(os.path.join(cache_dir, f"{i}.pt"))
        out_q.put(act.share_memory_())
    out_q.put("DONE")


def new_wandb_process(
    config,
    log_queue,
    entity,
    project,
    run_id_queue=None,
    index=None,
):
    run = wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    if run_id_queue is not None:
        run_id_queue.put((index, run.id))
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            if isinstance(log, dict):
                wandb.log(log)
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
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def train(
    q,
    trainer_configs,
    run_cfg,
    use_wandb,
    wandb_entity,
    wandb_project,
    save_dir,
    log_steps,
    verbose,
    save_steps,
    max_errors: int = MAX_ERRORS,
    error_queue: mp.Queue = None,
    run_id_queue: Optional[mp.Queue] = None,
):
    faulthandler.enable()
    wandb_processes = []
    log_queues = []

    print("Creating trainers...")
    trainers = []
    for cfg in trainer_configs:
        cls = cfg.pop("trainer")
        trainers.append(cls(**cfg))

    if use_wandb:
        print("Starting wandb processes...")
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue(32)
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {
                k: v.cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in wandb_config.items()
            }
            wandb_process = mp.Process(
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
            wandb_process.start()
            wandb_processes.append(wandb_process)

    if save_dir is not None:
        print("Creating save directories...")
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, trainer_dir in zip(trainers, save_dirs):
            os.makedirs(trainer_dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            with open(os.path.join(trainer_dir, "config.json"), "w") as f:
                print(config)
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    print("Starting training...")
    step = 0
    errors = 0
    while True:
        try:
            act = q.get(timeout=1)
        except queue.Empty:
            continue
        if act == "DONE":
            break
        try:
            for t in trainers:
                if (use_wandb or verbose) and step % log_steps == 0:
                    log_stats(
                        trainers,
                        step,
                        act,
                        False,  # activations_split_by_head, TODO: re-enable
                        False,  # transcoder, TODO: re-enable
                        log_queues=log_queues,
                        verbose=verbose,
                    )

                if save_steps is not None and step in save_steps:
                    for idx, (trainer_dir, trainer) in enumerate(zip(save_dirs, trainers)):
                        if trainer_dir is not None:
                            
                            # TODO: re-enable
                            # if normalize_activations:
                            #     # Temporarily scale up biases for checkpoint saving
                            #     trainer.ae.scale_biases(norm_factor)

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

                            # TODO: re-enable
                            # if normalize_activations:
                            #     trainer.ae.scale_biases(1 / norm_factor)

                for trainer in trainers:
                    trainer.update(step, act)
            step += 1
        except Exception:
            errors += 1
            tb = traceback.format_exc()
            if error_queue is not None:
                error_queue.put(("train", tb))
            traceback.print_exc()
            if errors >= max_errors:
                print("train: too many errors, exiting")
                if error_queue is not None:
                    error_queue.put(("train", "too many errors"))
                sys.exit(1)
            else:
                print("train: error encountered, skipping batch")
                continue

    # Signal wandb processes to finish
    if use_wandb:
        for wnb_q in log_queues:
            wnb_q.put("DONE")
        for process in wandb_processes:
            process.join()


def _run_single_process(
    trainer_configs,
    *,
    device="cuda:0",
    model_name="EleutherAI/pythia-70m-deduped",
    submodule=None,
    dataset_name="HuggingFaceFW/fineweb",
    steps=100_000,
    batch_size=128,
    seq_len=32,
    activation_cache_dir=None,
    run_cfg={},
    use_wandb=False,
    wandb_entity=None,
    wandb_project=None,
    save_dir=None,
    log_steps=100,
    verbose=False,
    save_steps=None,
    max_errors: int = MAX_ERRORS,
    run_id_queue: Optional[mp.Queue] = None,
):
    """Run the entire pipeline in a single process."""
    faulthandler.enable()

    cache_exists = (
        activation_cache_dir is not None
        and os.path.exists(os.path.join(activation_cache_dir, f"{steps-1}.pt"))
    )

    if not cache_exists:
        dataset = load_dataset(dataset_name, streaming=True, split="train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            prefetch_factor=8,
            persistent_workers=True,
        )

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.backend_tokenizer.enable_truncation(max_length=seq_len)
        tok.backend_tokenizer.enable_padding(
            length=seq_len, pad_id=tok.pad_token_id, pad_token=tok.pad_token
        )

        model = LanguageModel(model_name, device_map=device)
        model.eval()
        submodule_ref = eval(f"model.{submodule}")
        iterator = iter(loader)
        if activation_cache_dir is not None:
            os.makedirs(activation_cache_dir, exist_ok=True)

    # Setup trainers and wandb processes
    trainers = []
    wandb_processes = []
    log_queues = []
    if use_wandb and run_id_queue is None:
        run_id_queue = mp.Queue()
    for cfg in trainer_configs:
        cls = cfg.pop("trainer")
        trainers.append(cls(**cfg))

    for trainer in trainers:
        trainer.ae.to(device)

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

    if cache_exists:
        iterator = None
    else:
        iterator = iter(loader)
    errors = 0
    step = 0
    with tqdm(total=steps, desc="Training...") as pbar:
        while step < steps:
            try:
                if cache_exists:
                    act = torch.load(
                        os.path.join(activation_cache_dir, f"{step}.pt"),
                        map_location="cpu",
                    )
                else:
                    batch = next(iterator)
                    inputs = tok(
                        batch["text"],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=seq_len,
                        truncation=True,
                    )
                    inputs = {k: v.to("cpu", non_blocking=True) for k, v in inputs.items()}

                    with torch.inference_mode():
                        with model.trace(inputs, invoker_args={"max_length": seq_len}):
                            h = submodule_ref.output.save()
                            submodule_ref.output.stop()
                        try:
                            act = h.value.to("cpu", non_blocking=True)
                        except AttributeError:
                            act = h.value[0].to("cpu", non_blocking=True)
                    if activation_cache_dir is not None:
                        torch.save(act, os.path.join(activation_cache_dir, f"{step}.pt"))

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
                            checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                            path = os.path.join(trainer_dir, "checkpoints", f"ae_{step}.pt")
                            torch.save(
                                checkpoint,
                                path,
                            )
                            if use_wandb:
                                log_queues[idx].put(("artifact", path))

                    tnr.update(step, act)

                step += 1
                pbar.update(1)
            except Exception:
                errors += 1
                traceback.print_exc()
                if errors >= max_errors:
                    raise
                else:
                    continue

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

def run_pipeline(
    trainer_configs,
    *,
    device="cuda:0",
    model_name="EleutherAI/pythia-70m-deduped",
    submodule=None,
    layer=3,
    activation_dim=512,
    dictionary_size=None,
    dataset_name="HuggingFaceFW/fineweb",
    steps=100_000,
    batch_size=128,
    seq_len=32,
    activation_cache_dir=None,
    queue_size=32,
    run_cfg={},
    use_wandb=False,
    wandb_entity=None,
    wandb_project=None,
    save_dir=None,
    log_steps=100,
    verbose=False,
    save_steps=None,
    max_errors: int = MAX_ERRORS,
    start_method: str = "forkserver",
    multiprocess: bool = True,
):
    if dictionary_size is None:
        dictionary_size = 16 * activation_dim

    mp.set_start_method(start_method, force=True)  # avoids CUDA fork issues
    run_id_queue = mp.Queue() if use_wandb else None

    if multiprocess:
        error_q = mp.Queue()
        use_cache = (
            activation_cache_dir is not None
            and os.path.exists(os.path.join(activation_cache_dir, f"{steps-1}.pt"))
        )

        if use_cache:
            act_q = mp.Queue(queue_size)
            procs = [
                mp.Process(
                    name="activation_loader",
                    target=activation_loader,
                    args=(act_q, activation_cache_dir, steps),
                ),
                mp.Process(
                    name="train",
                    target=train,
                    args=(
                        act_q,
                        trainer_configs,
                        run_cfg,
                        use_wandb,
                        wandb_entity,
                        wandb_project,
                        save_dir,
                        log_steps,
                        verbose,
                        save_steps,
                        max_errors,
                        error_q,
                        run_id_queue,
                    ),
                ),
            ]
        else:
            in_q, act_q = mp.Queue(queue_size), mp.Queue(queue_size)
            if activation_cache_dir is not None:
                os.makedirs(activation_cache_dir, exist_ok=True)
            procs = [
                mp.Process(
                    name="input_fetcher",
                    target=input_fetcher,
                    args=(
                        in_q,
                        model_name,
                        dataset_name,
                        steps,
                        batch_size,
                        seq_len,
                        device,
                        max_errors,
                        error_q,
                    ),
                ),
                mp.Process(
                    name="activation_fetcher",
                    target=activation_fetcher,
                    args=(
                        in_q,
                        act_q,
                        model_name,
                        submodule,
                        seq_len,
                        device,
                        activation_cache_dir,
                        max_errors,
                        error_q,
                    ),
                ),
                mp.Process(
                    name="train",
                    target=train,
                    args=(
                        act_q,
                        trainer_configs,
                        run_cfg,
                        use_wandb,
                        wandb_entity,
                        wandb_project,
                        save_dir,
                        log_steps,
                        verbose,
                        save_steps,
                        max_errors,
                        error_q,
                        run_id_queue,
                    ),
                ),
            ]

        for p in procs:
            p.start()

        alive = procs.copy()
        try:
            while alive:
                try:
                    while True:
                        proc_name, tb = error_q.get_nowait()
                        print(f"[Error in {proc_name}]", file=sys.stderr)
                        print(tb, file=sys.stderr)
                except queue.Empty:
                    pass

                for p in list(alive):
                    p.join(timeout=0.1)
                    if not p.is_alive():
                        if p.exitcode != 0:
                            raise RuntimeError(
                                f"Process {p.name} exited with code {p.exitcode}"
                            )
                        alive.remove(p)
        finally:
            for p in alive:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()
        if use_wandb:
            run_ids = [None] * len(trainer_configs)
            for _ in range(len(trainer_configs)):
                idx, r_id = run_id_queue.get()
                run_ids[idx] = r_id
        else:
            run_ids = []
    else:
        run_ids = _run_single_process(
            trainer_configs,
            device=device,
            model_name=model_name,
            submodule=submodule,
            dataset_name=dataset_name,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            run_cfg=run_cfg,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            save_dir=save_dir,
            log_steps=log_steps,
            verbose=verbose,
            save_steps=save_steps,
            max_errors=max_errors,
            run_id_queue=run_id_queue,
            activation_cache_dir=activation_cache_dir,
        )
    return run_ids


def main() -> None:
    """Run the default training pipeline."""
    steps = 100_000
    device = "cuda"
    run_pipeline(
        [
            dict(
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                activation_dim=512,
                dict_size=8192,
                lr=1e-3,
                device=device,
                steps=steps,
                layer=3,
                lm_name="EleutherAI/pythia-70m-deduped",
                warmup_steps=100,
                sparsity_warmup_steps=100,
            ),
        ],
        submodule="gpt_neox.layers[3].mlp",
        steps=steps,
        batch_size=128,
        use_wandb=False,
        device=device,
        start_method="forkserver",
    )


if __name__ == "__main__":
    main()
