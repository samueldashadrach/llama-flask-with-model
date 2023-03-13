# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


#imports beyond facebook repo
import random
import os
import string

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

# following code runs when example.py starts

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")
gen_global = load(
    ckpt_dir = "../weights/7B",
    tokenizer_path = "../weights/tokenizer.model",
    local_rank = local_rank,
    world_size = world_size,
    max_seq_len = 512,
    max_batch_size = 32
)

# infinite loop
while True:

    # do I need to add a waiting period here?? This program runs using torchrun and is hence parallelised

    try:
        with open("prompt", "r") as f_prompt:
            # wait for "prompt" file to be created, then read and destroy "prompt" file
            prompt = f_prompt.read()
            f_prompt.close()
            os.remove("prompt")

            result = gen_global.generate(
                prompt, max_gen_len=256, temperature=0.8, top_p=0.95
            )
            console.log(prompt)
            console.log(result)

            # write to "result" file
            tempname = "".join(random.choices(string.ascii_uppercase, k=20))
            try:
                with open(tempname, "w") as f_result_temp:
                    f_result_temp.write(result)
                    f_result_temp.close()
                    os.rename(tempname, "result")

            except IOError:
                print("Result could not be written!!!!")
    except IOError:
        pass

    