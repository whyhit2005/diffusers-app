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
    return parser.parse_args(input_args)


def load_config_yaml(args):
    with open(args.cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["infer"])
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
    if cfg_args.add_vae:
        vae = AutoencoderKL.from_pretrained(
            cfg_args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)
        pipe.vae = vae
        
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
    return pipe


def infer(model_dir, sample_dir, prompt_list, cfg_args):
    pipe = load_base_model(cfg_args)
    pipe.load_lora_weights(
        str(model_dir), 
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="custom")
    pipe.fuse_lora(lora_scale=1.0)
    
    # pipe.load_lora_weights(
    #     "/home/wangyh/sdxl_models/lora/", 
    #     weight_name="mengwa.safetensors",
    #     adapter_name="extra")
    # pipe.load_lora_weights(
    #     "/home/wangyh/sdxl_models/lora/", 
    #     weight_name="CLAYMATE_V2.03_.safetensors",
    #     adapter_name="extra")
    # pipe.set_adapters(["custom", "extra"], adapter_weights=[1.0, 1.0])
    # pipe.fuse_lora(adapter_names=["custom", "extra"], lora_scale=1.0)
    
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    embedding_path = model_dir/"pytorch_lora_emb.safetensors"
    if not embedding_path.exists():
        embedding_path = model_dir / ".." / "pytorch_lora_emb.safetensors"
    state_dict = load_file(str(embedding_path))
    # load embeddings of text_encoder 1 (CLIP ViT-L/14)
    pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    # load embeddings of text_encoder 2 (CLIP ViT-G/14)
    pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
    
    neg_emb_path = Path("/home/wangyh/sdxl_models/embedding/") / "negativeXL_D.safetensors"
    state_dict = load_file(str(neg_emb_path))
    neg_tokens = []
    negative_prompt = ""
    if cfg_args.negative_prompt:
        negative_prompt += f"{cfg_args.negative_prompt}, "
    for i in range(state_dict["clip_l"].shape[0]):
        neg_tokens.append(f"<n{i}>")
    pipe.load_textual_inversion(state_dict["clip_l"], token=neg_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    pipe.load_textual_inversion(state_dict["clip_g"], token=neg_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
    negative_prompt += " ".join(neg_tokens)
    
    num_sample = cfg_args.sample_num
    logger.info(f"Model:{sample_dir.parent.name} {model_dir.name}")
    generator = torch.Generator("cuda").manual_seed(cfg_args.seed)
    for i, item in enumerate(prompt_list):
        prompt = item
        logger.info(f"{i} Prompt: {prompt}")
        images = pipe(
            prompt=prompt, 
            # negative_prompt=negative_prompt,
            num_inference_steps=cfg_args.infer_steps, 
            num_images_per_prompt = cfg_args.sample_num,
            guidance_scale = 5.0,
            # cross_attention_kwargs={"scale": 1.0},
            height = 1024, width = 1024,
            generator=generator,
        ).images
        for j, image in enumerate(images):
            sample_path = sample_dir / f"{model_dir.name}-p{i:03d}-{j:03d}.png"
            image.save(str(sample_path))
    
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    pipe.unload_textual_inversion()
    del pipe


def main(cfg_args):
    if cfg_args.work_dir is None:
        raise ValueError("work directory is not set")
    prompt_list = []
    if cfg_args.prompt_file is not None:
        with open(cfg_args.prompt_file, "r") as f:
            for line in f:
                line = line.strip()
                if cfg_args.replace_word:
                    replace_to = f'{cfg_args.token_abstraction} {cfg_args.class_prompt}'
                    line = line.replace(cfg_args.replace_word, replace_to)
                prompt_list.append(line)
    else:
        prompt_list = [cfg_args.prompt]

    work_dir = Path(cfg_args.work_dir)
    dirlist = None
    if cfg_args.subfolders:
        dirlist = list(work_dir.iterdir())
        dirlist = sorted(dirlist, key=lambda x: x.name)
        if cfg_args.sub_range:
            s, e = cfg_args.sub_range.split("-")
            s, e = int(s), int(e)
            s = max(0, s)
            e = min(len(dirlist), e)
            dirlist = dirlist[s:e]
    else:
        dirlist = [work_dir]
    dirlist = sorted(dirlist, key=lambda x: x.name)
    for cdir in dirlist:
        instance_token = cdir.name.split("-")[-1]
        if cfg_args.task_name:
            model_dir = cdir / f"{cfg_args.model_dir_name}-{cfg_args.task_name}"
        else:
            model_dir = cdir / cfg_args.model_dir_name
        if cfg_args.checkpoint is not None:
            checkpoint_dirs = list(model_dir.glob("checkpoint*"))
            checkpoint_dirs = sorted(checkpoint_dirs)
            start_i = cfg_args.checkpoint % len(checkpoint_dirs)
            model_dirs = checkpoint_dirs[start_i:]
            model_dirs = sorted(model_dirs, key=lambda x:x.name)
        else:
            model_dirs = [model_dir]
        
        if cfg_args.task_name:
            sample_dir = cdir / f"{cfg_args.sample_dir_name}-{cfg_args.task_name}"
        else:
            sample_dir = cdir / cfg_args.sample_dir_name
        if cfg_args.init_new:
            shutil.rmtree(str(sample_dir), ignore_errors=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for mdir in model_dirs:
            infer(mdir, sample_dir, prompt_list, cfg_args)
    return 0


if __name__ == "__main__":
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = main(cfg_args)
    sys.exit(ret)