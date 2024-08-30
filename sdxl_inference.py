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
                        default=3,
                        help="sample number")
    parser.add_argument("--infer_steps", type=int,
                        default=50,
                        help="inference steps")
    parser.add_argument("--subfolders", action="store_true",
                        default=False,
                        help="use subfolder")
    parser.add_argument("--sub_range", type=str,
                        default=None,
                        help="subfolder range")
    parser.add_argument("--checkpoint", type=int,
                        default=-1,
                        help="use checkpoint from i")
    parser.add_argument("--init_new", action="store_true",
                        default=False,
                        help="initialize new folder")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="pretrained model")
    parser.add_argument("--no_add_vae", action="store_true",
                    default=False,
                    help="no add vae")
    parser.add_argument("--from_file", type=str,
                        default=None,
                        help="from file")
    return parser.parse_args(input_args)

def load_base_model(args):
    if Path(args.pretrained_model_name_or_path).is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.pretrained_model_name_or_path,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            # vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    if not args.no_add_vae:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe.vae = vae
        
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
    return pipe

def infer(model_dir, sample_dir, prompt_list, args):
    pipe = load_base_model(args)
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
    for i in range(state_dict["clip_l"].shape[0]):
        neg_tokens.append(f"<n{i}>")
    pipe.load_textual_inversion(state_dict["clip_l"], token=neg_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    pipe.load_textual_inversion(state_dict["clip_g"], token=neg_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
    negative_prompt = "".join(neg_tokens)
    if args.negative_prompt:
        negative_prompt += f"{args.negative_prompt}"
        
    num_sample = args.sample_num
    logger.info(f"Model:{sample_dir.parent.name} {model_dir.name}")
    for i, item in enumerate(prompt_list):
        prompt = item["prompt"]
        seed = item["seed"]
        generator = torch.Generator("cuda").manual_seed(seed)
        logger.info(f"{i} Prompt: {prompt}")
        images = pipe(
                prompt=prompt, 
                # negative_prompt=negative_prompt,
                num_inference_steps=args.infer_steps, 
                num_images_per_prompt = args.sample_num,
                guidance_scale = 3.0,
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
    
def main(args):
    random.seed(0)
    prompt_list = []
    if args.from_file:
        with open(args.from_file, "r") as f:
            for line in f:
                line = line.strip()
                t = json.loads(line)
                if "seed" not in t:
                    t["seed"] = random.randint(int(1e6), int(1e7))
                prompt_list.append(t)
    else:
        prompt_list = [{"prompt": args.prompt, "seed": random.randint(int(1e6), int(1e7))}]

    work_dir = Path(args.work_dir)
    dirlist = None
    if args.subfolders:
        dirlist = list(work_dir.iterdir())
        dirlist = sorted(dirlist, key=lambda x: x.name)
        if args.sub_range:
            s, e = args.sub_range.split("-")
            s, e = int(s), int(e)
            s = max(0, s)
            e = min(len(dirlist), e)
            dirlist = dirlist[s:e]
    else:
        dirlist = [work_dir]
    dirlist = sorted(dirlist, key=lambda x: x.name)
    for cdir in dirlist:
        instance_token = cdir.name.split("-")[-1]
        model_dir = cdir / "models"
        model_dirs = [model_dir]
        checkpoint_dir = sorted(list(model_dir.glob("checkpoint*")))
        if args.checkpoint > 0 and args.checkpoint < len(checkpoint_dir):
            model_dirs = checkpoint_dir[args.checkpoint:]
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