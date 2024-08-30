import torch
from pathlib import Path
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
import os, sys, shutil
from IPython.display import Image, display
import argparse
from tqdm import tqdm
import json

def read_prompts(prompt_file):
    pdata = []
    pdict = {}
    with open(prompt_file, "r") as f:
        for line in f:
            start = line.find("Photo of a")
            lstr = line[start:]
            if lstr not in pdict:
                pdict[lstr] = 1
                pdata.append(lstr)
    tdata = []
    tdict = {}
    for lstr in pdata:
        words = lstr.split()
        for word in words:
            if word not in tdict:
                tdict[word] = 0
            tdict[word] += 1
    
    for lstr in pdata:
        words = lstr.split()
        for word in words:
            if tdict[word] < 4:
                tdata.append(lstr)
                break
    pdata = tdata
    return pdata

def infer():
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    # pretrained_model_name_or_path = "/home/wangyh/sdxl_models/checkpoint/juggernautXL_v9Rundiffusionphoto2.safetensors"
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLPipeline.from_single_file(
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    _ = pipe.to("cuda")
    pipe.disable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    
    generator = torch.Generator("cuda").manual_seed(5678)
    prompt_file = "woman-prompts.txt"
    # pdata = read_prompts(prompt_file)
    pdata = range(200)
    sample_dir = Path("regular-samples")
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    bar = tqdm(desc="Generating samples", total=len(pdata))
    for i, item in enumerate(pdata):
        bar.update(1)
        prompt = "photo of a person, solo, look at viewer, full body"
        # negative_prompt = "".join(neg_tokens)
        negative_prompt = "painting, drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label"
        simg = pipe(
            prompt=prompt, 
            # negative_prompt=negative_prompt,
            num_inference_steps=30,
            num_images_per_prompt = 1,
            guidance_scale = 9.0,
            # cross_attention_kwargs={"scale": 1.0},
            height = 1024, width = 1024,
            # generator=generator,
        ).images[0]
        image_path = sample_dir / f"{i:03d}.png"
        cap_path = sample_dir / f"{i:03d}.txt"
        simg.save(str(image_path), "PNG", quality=100)
        # with open(cap_path, "w") as f:
        #     f.write(prompt)

if __name__ == "__main__":
    infer()