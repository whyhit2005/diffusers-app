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
import random
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--work_dir", type=str,
                        default=None, required=True,
                        help="images, model, working path")
    parser.add_argument("--prompt", type=str,
                        default=None,
                        help="validation prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default=None,
                        help="negative prompt")
    parser.add_argument("--sample_dir", type=str,
                        default="samples",
                        help="sample directory")
    parser.add_argument("--sample_num", type=int,
                        default=30,
                        help="sample number")
    parser.add_argument("--infer_steps", type=int,
                        default=50,
                        help="inference steps")
    parser.add_argument("--subfolders", action="store_true",
                        default=False,
                        help="use subfolder")
    parser.add_argument("--init_new", action="store_true",
                        default=False,
                        help="initialize new folder")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="pretrained model")
    parser.add_argument("--checkpoint", action="store_true",
                        default=False,
                        help="use checkpoint")
    parser.add_argument("--from_file", type=str,
                        default=None,
                        help="from file")
    return parser.parse_args(input_args)

def infer(model_dir, sample_dir, prompt_list, args):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    if Path(args.pretrained_model_name_or_path).is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        str(model_dir), 
        weight_name="lora_weights.safetensors",
        adapter_name="custom")
    pipe.fuse_lora(lora_scale=1.0)

    num_sample = args.sample_num
    pipe.enable_xformers_memory_efficient_attention()

    logger.info(f"Model: {model_dir.name}")
    for i, item in enumerate(prompt_list):
        prompt = item["prompt"]
        seed = item["seed"]
        generator = torch.Generator("cuda").manual_seed(seed)
        logger.info(f"{i} Prompt: {prompt}")
        for j in range(num_sample):
            simg = pipe(
                    prompt=prompt, 
                    # negative_prompt=negative_prompt,
                    num_inference_steps=args.infer_steps, 
                    num_images_per_prompt = 1,
                    guidance_scale = 10.0,
                    # cross_attention_kwargs={"scale": 1.0},
                    height = 1024, width = 1024,
                    generator=generator,
            ).images[0]
            if args.checkpoint:
                sample_path = sample_dir / f"{model_dir.name}-p{i:03d}-{j:03d}.png"
            else:
                sample_path = sample_dir / f"p{i:03d}-{j:03d}.png"
            simg.save(str(sample_path))

def main(args):
    random.seed(0)
    prompt_list = []
    if args.from_file:
        with open(args.from_file, "r") as f:
            for line in f:
                seed = random.randint(int(1e5), int(1e6))
                prompt_list.append({"prompt": line.strip(), "seed": seed})
    else:
        prompt_list.append({"prompt": args.prompt, "seed": random.randint(int(1e5), int(1e6))})
    
    work_dir = Path(args.work_dir)
    dirlist = None
    if args.subfolders:
        dirlist = list(work_dir.iterdir())
    else:
        dirlist = [work_dir]
    for cdir in dirlist:
        instance_token = cdir.name.split("-")[-1]
        for prompt in prompt_list:
            prompt["prompt"] = prompt["prompt"].replace("<s0><s1>", instance_token)
        model_dir = cdir / "models"
        model_dirs = [model_dir]
        if args.checkpoint:
            model_dirs = list(model_dir.glob("checkpoint*"))
        model_dirs = sorted(model_dirs, key=lambda x:x.name)
        
        sample_dir = cdir / args.sample_dir
        if args.init_new:
            shutil.rmtree(str(sample_dir), ignore_errors=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for mdir in model_dirs:
            infer(mdir, sample_dir, prompt_list, args)
    return 0

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))