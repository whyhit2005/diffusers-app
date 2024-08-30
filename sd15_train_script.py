import os, sys, shutil
from tqdm import tqdm
from pathlib import Path
import random
import json
import argparse
import warnings
import datetime
import logging

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--work_dir", type=str,
                        default="", required=True,
                        help="images, model, working path")
    parser.add_argument("--subfolders", action="store_true",
                        default=False,
                        help="process subfolders")
    parser.add_argument("--max_train_steps", type=int,
                        default=1000,
                        help="max train steps")
    parser.add_argument("--validation_prompt", type=str,
                        default=None,
                        help="validation prompt suffix")
    parser.add_argument("--init_new", action="store_true",
                        default=False,
                        help="initialize new training")
    parser.add_argument("--train_batch_size", type=int,
                        default=10,
                        help="batch size")
    parser.add_argument("--rank", type=int,
                        default=64,
                        help="rank")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="pretrained model")
    parser.add_argument("--use_adamw", action="store_true",
                        default=False,
                        help="use adamw optimizer")
    
    return parser.parse_args(input_args)

def train_process(args):
    logging.basicConfig(level=logging.INFO)
    work_dir = Path(args.work_dir)
    dirlist = []
    if args.subfolders:
        dirlist = list(work_dir.iterdir())
    else:
        dirlist = [work_dir]
    dirlist = sorted(dirlist, key=lambda x: x.name)

    for tdir in dirlist:
        instance = tdir.name.split("-")[-1]
        token_abstraction = instance
        model_dir = tdir / "models"
        images_dir = tdir / "images"
        logging_dir = tdir / "logs"
        
        if args.init_new and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if args.init_new and os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        cmdfile = tdir / "cmd.txt"
        donefile = model_dir / "done.txt"
        if os.path.exists(donefile) and model_dir.glob("*safetensors"):
            logging.info(f"{instance} already done, skip")
            continue
        
        instant_prompt = ""
        metafile = os.path.join(images_dir, "metadata.jsonl")
        with open(metafile, "r") as f:
            lines = f.readlines()
            prompt = json.loads(lines[0])["prompt"]
            instant_prompt = prompt.split(",")[0]
            
        if args.validation_prompt and instant_prompt:
            validation_prompt = f'{instant_prompt}, {args.validation_prompt}'
            validation_epochs = args.num_train_epochs // 10
        else:
            validation_prompt = None
            validation_epochs = 0
        
        cmdstr = f'accelerate launch train_dreambooth_lora_sd15_advanced.py'
        cmdstr += f' --pretrained_model_name_or_path=\"{args.pretrained_model_name_or_path}\"'
        cmdstr += f' --pretrained_vae_model_name_or_path="stabilityai/sd-vae-ft-mse"'
        cmdstr += f' --dataset_name={images_dir}'
        cmdstr += f' --instance_prompt=\"{instant_prompt}\"'
        cmdstr += f' --token_abstraction=\"{token_abstraction}\"'
        if validation_prompt:
            cmdstr += f' --validation_prompt=\"{validation_prompt}\"'
            cmdstr += f' --validation_epochs={validation_epochs}'
        cmdstr += f' --output_dir={model_dir}'
        cmdstr += f' --logging_dir={logging_dir}'
        cmdstr += f' --caption_column="prompt"'
        cmdstr += f' --mixed_precision="bf16"'
        cmdstr += f' --resolution=512'
        cmdstr += f' --train_batch_size={args.train_batch_size}'
        cmdstr += f' --repeats={args.train_batch_size}'
        cmdstr += f' --report_to="wandb"'
        cmdstr += f' --gradient_accumulation_steps=1'
        cmdstr += f' --gradient_checkpointing'
        if args.use_adamw:
            cmdstr += f' --learning_rate=2e-4'
            cmdstr += f' --text_encoder_lr=2e-5'
            cmdstr += f' --optimizer="adamw"'
            cmdstr += f' --lr_scheduler="cosine"'
            cmdstr += f' --lr_warmup_steps=0'
        else:
            cmdstr += f' --learning_rate=1.0'
            cmdstr += f' --text_encoder_lr=1.0'
            cmdstr += f' --optimizer="prodigy"'
            cmdstr += f' --prodigy_safeguard_warmup=True'
            cmdstr += f' --prodigy_use_bias_correction=True'
            cmdstr += f' --lr_scheduler="cosine"'
            cmdstr += f' --lr_warmup_steps=0'
        cmdstr += f' --adam_weight_decay=0.01'
        cmdstr += f' --adam_beta1=0.9 --adam_beta2=0.99'
        cmdstr += f' --train_text_encoder_ti'
        cmdstr += f' --train_text_encoder_ti_frac=0.5'
        cmdstr += f' --snr_gamma=5.0'
        cmdstr += f' --rank={args.rank}'
        cmdstr += f' --max_train_steps={args.max_train_steps}'
        cmdstr += f' --checkpoints_total_limit=10'
        cmdstr += f' --checkpointing_steps=100'
        cmdstr += f' --resume_from_checkpoint=latest'
        cmdstr += f' --enable_xformers_memory_efficient_attention'
        cmdstr += f' --seed=42'
        
        with open(cmdfile, "w") as f:
            f.write(cmdstr)
        start_time = datetime.datetime.now()
        ret = os.system(cmdstr)
        if ret != 0:
            warnings.warn(f"{instance} Lora training failed")
            return ret
        end_time = datetime.datetime.now()
        with open(donefile, "w") as f:
            f.write("done\n")
            f.write(f"start time: {start_time}\n")
            f.write(f"end time: {end_time}\n")
            f.write(f"duration: {end_time-start_time}\n")
            f.write(f"cmd: {cmdstr}\n")
    return 0

def infer_after(args):
    work_dir = args.work_dir
    dirlist = []
    if args.subset:
        dirlist = args.subset.split(",")
    else:
        dirlist = os.listdir(work_dir)
    dirlist = sorted(dirlist)
    for instance in dirlist:
        sub_dir = os.path.join(work_dir, instance)
        if not os.path.exists(sub_dir):
            warnings.warn(f"Directory {sub_dir} does not exist")
            continue

        cmdstr = f'python inference.py'
        cmdstr += f' --work_dir={sub_dir}'
        if args.pretrained_model_name_or_path:
            cmdstr += f' --pretrained_model_name_or_path="{args.pretrained_model_name_or_path}"'
        print(f"{cmdstr}")
        ret = os.system(cmdstr)
    return 0

if __name__ == "__main__":
    args = parse_args()
    ret = train_process(args)
    # if ret == 0:
    #     ret = infer_after(args)
    sys.exit(ret)
