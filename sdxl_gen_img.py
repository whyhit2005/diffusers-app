import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
import os, sys, shutil
from IPython.display import Image, display
import argparse
from tqdm import tqdm
import warnings
import random, json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
from omegaconf import OmegaConf
import yaml


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="sdxl text to images script.")
    parser.add_argument("cfg_file", type=str,
                        help="config file")
    parser.add_argument("--sample_num", type=int, default=5,
                        help="number of samples per prompt")
    parser.add_argument("--infer_steps", type=int, default=40,
                        help="number of inference steps")
    parser.add_argument("--seed", type=int, default=666666,
                        help="random seed")
    return parser.parse_args(input_args)


def load_config_yaml(args):
    with open(args.cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["train"])
        # t_data["infer"] = cfg_data["infer"]
        cfg_args = OmegaConf.create(t_data)
        return cfg_args


def load_base_model(cfg_args):
    if Path(cfg_args.pretrained_model_name_or_path).is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            cfg_args.pretrained_model_name_or_path,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg_args.pretrained_model_name_or_path,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    logger.info(f"Loaded model: {cfg_args.pretrained_model_name_or_path}")
    if cfg_args.add_vae:
        if Path(cfg_args.pretrained_vae_model_name_or_path).is_file():
            vae = AutoencoderKL.from_single_file(
                cfg_args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)
        else:
            vae = AutoencoderKL.from_pretrained(
                cfg_args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)
        pipe.vae = vae
        logger.info(f"Loaded VAE: {cfg_args.pretrained_vae_model_name_or_path}")
    pipe = pipe.to("cuda")
    # pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
    return pipe


def infer(pipe, sample_dir, prompt_list, cfg_args, args):
    num_sample = args.sample_num
    steps = args.infer_steps
    generator = torch.Generator("cuda").manual_seed(args.seed)
    for i, item in enumerate(prompt_list):
        prompt = item
        logger.info(f"{i} Prompt: {prompt}")
        images = pipe(
            prompt=prompt, 
            num_inference_steps=steps, 
            num_images_per_prompt = num_sample,
            guidance_scale = 7.0,
            height = 1024, width = 1024,
            generator=generator,
        ).images
        for j, image in enumerate(images):
            sample_path = sample_dir / f"p{i:03d}-{j:03d}.png"
            image.save(str(sample_path))
    del pipe


def read_prompt_file(meta_file, cfg_args):
    prompt_list = []
    with open(meta_file, "r") as f:
        for line in f:
            line = line.strip()
            json_data = json.loads(line)
            prompt = json_data["prompt"]
            # prompt = prompt.replace(cfg_args.token_abstraction, "")
            prompt_list.append(prompt)
    return prompt_list


def copy_image(meta_file, src_dir, tgt_dir):
    with open(meta_file, "r") as f:
        i = 0
        for line in f:
            line = line.strip()
            json_data = json.loads(line)
            image_name = json_data["file_name"]
            src_path = src_dir / image_name
            tgt_path = tgt_dir / f"p{i:03d}{image_name}"
            shutil.copyfile(str(src_path), str(tgt_path))
            i += 1

    
def main(cfg_args, args):
    if cfg_args.work_dir is None:
        raise ValueError("work directory is not set")

    pipe = load_base_model(cfg_args)
    work_dir = Path(cfg_args.work_dir)
    dirlist = None
    if cfg_args.subfolders:
        dirlist = list(work_dir.iterdir())
        dirlist = sorted(dirlist, key=lambda x: x.name)
    else:
        dirlist = [work_dir]
    dirlist = sorted(dirlist, key=lambda x: x.name)
    for cdir in dirlist:
        image_dir = cdir / cfg_args.images_dir_name
        meta_file = image_dir / "metadata.jsonl"
        prompt_list = read_prompt_file(meta_file, cfg_args)
        sample_dir = cdir / f"caption-samples"
        if cfg_args.init_new:
            shutil.rmtree(str(sample_dir), ignore_errors=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        copy_image(meta_file, image_dir, sample_dir)
        infer(pipe, sample_dir, prompt_list, cfg_args, args)
    del pipe
    return 0


if __name__ == "__main__":
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = main(cfg_args, args)
    sys.exit(ret)