import json
import multiprocessing as mp
import os
import queue

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


def input_fetcher(q, model_name, dataset_name, steps, batch_size, max_length, device):
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
    with tqdm(total=steps, desc="Training...") as pbar:
        for i, batch in enumerate(loader):
            if i >= steps:
                break
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
    q.put("DONE")


def activation_fetcher(in_q, out_q, model_name, submodule, ctx_len, device):
    model = LanguageModel(model_name, device_map=device)
    model.eval()
    submodule = eval(f"model.{submodule}")  # caller decides string

    while True:
        try:
            inputs = in_q.get(timeout=1)
            if inputs == "DONE":
                break
            with torch.inference_mode():
                with model.trace(inputs, invoker_args={"max_length": ctx_len}):
                    h = submodule.output.save()
                    submodule.output.stop()
                out_q.put(h.value.to("cpu", non_blocking=True).share_memory_())
        except queue.Empty:
            continue
    out_q.put("DONE")


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
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
                # fraction of variance explained
                total_variance = torch.var(act, dim=0).sum()
                residual_variance = torch.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

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
):
    wandb_processes = []
    log_queues = []

    print("Creating trainers...")
    trainers = []
    for cfg in trainer_configs:
        cls = cfg.pop("trainer")
        trainers.append(cls(**cfg))

    if use_wandb:
        print("Starting wandb processes...")
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
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
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    if save_dir is not None:
        print("Creating save directories...")
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            with open(os.path.join(dir, "config.json"), "w") as f:
                print(config)
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    print("Starting training...")
    step = 0
    while True:
        try:
            act = q.get(timeout=1)
            if act == "DONE":
                break
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
                    for dir, trainer in zip(save_dirs, trainers):
                        if dir is not None:

                            # TODO: re-enable
                            # if normalize_activations:
                            #     # Temporarily scale up biases for checkpoint saving
                            #     trainer.ae.scale_biases(norm_factor)

                            if not os.path.exists(os.path.join(dir, "checkpoints")):
                                os.mkdir(os.path.join(dir, "checkpoints"))

                            checkpoint = {
                                k: v.cpu() for k, v in trainer.ae.state_dict().items()
                            }
                            torch.save(
                                checkpoint,
                                os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                            )

                            # TODO: re-enable
                            # if normalize_activations:
                            #     trainer.ae.scale_biases(1 / norm_factor)

                for trainer in trainers:
                    trainer.update(step, act)
            step += 1
        except queue.Empty:
            continue

    # Signal wandb processes to finish
    if use_wandb:
        for wnb_q in log_queues:
            wnb_q.put("DONE")
        for process in wandb_processes:
            process.join()


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
    queue_size=32,
    run_cfg={},
    use_wandb=False,
    wandb_entity=None,
    wandb_project=None,
    save_dir=None,
    log_steps=100,
    verbose=False,
    save_steps=None,
):
    if dictionary_size is None:
        dictionary_size = 16 * activation_dim

    mp.set_start_method("forkserver", force=True)  # avoids CUDA fork issues
    in_q, act_q = mp.Queue(queue_size), mp.Queue(queue_size)

    procs = [
        mp.Process(
            target=input_fetcher,
            args=(in_q, model_name, dataset_name, steps, batch_size, seq_len, device),
        ),
        mp.Process(
            target=activation_fetcher,
            args=(in_q, act_q, model_name, submodule, seq_len, device),
        ),
        mp.Process(
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
            ),
        ),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    steps = 100_000
    device = "mps"
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
    )
