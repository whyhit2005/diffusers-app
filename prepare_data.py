from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import json
import os,sys,shutil
from tqdm import tqdm
from PIL import Image
import gc
import argparse
import warnings
from pathlib import Path
import re
import wd14_tagger

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths

def read_data(instance_dir):
    # create a list of (Pil.Image, path) pairs
    if not instance_dir.exists():
        print(f"Directory {instance_dir} does not exist")
        raise FileNotFoundError

    imgs_and_paths = []
    filelist = glob_images_pathlib(instance_dir, recursive=False)
    for file in filelist:
        imgs_and_paths.append((Image.open(file), file))
    return imgs_and_paths

# load the processor and the captioning model
def blip_prepare_data(imgs_and_paths, output_dir, instance_prompt, caption_prefix):
    # captioning utility
    def caption_images(input_image):
        inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values

        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=60)
        generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_caption
    
    with open(output_dir / 'metadata.jsonl', 'w') as outfile:
        for image, path in tqdm(imgs_and_paths, desc=f"{instance_prompt}"):
            caption = f"{caption_prefix} {instance_prompt}"
            caption += ", "+caption_images(image).split("\n")[0]
            tarpath = output_dir / path.name
            shutil.copy(path, tarpath)
            entry = {"file_name":path.name, "prompt": caption}
            json.dump(entry, outfile)
            outfile.write('\n')


def wd14_prepare_data(imgs_and_paths, output_dir, instance_prompt, caption_prefix):
    wd14args.desc = instance_prompt
    tag_results = wd14_tagger.tag_images(wd14_model, wd14_all_tags, imgs_and_paths, wd14args)
    tag_black_list = [
    "eye", "lip", "nose", "ear", "mouth", "teeth", "tongue", "neck",
    "blurry",
    ]
    pstr = r"|".join(tag_black_list)
    pattern = re.compile(pstr, re.IGNORECASE)
    
    freqdict = {}
    for img_path, tags in tag_results:
        tokens = tags.split(", ")
        for token in tokens:
            if token not in freqdict:
                freqdict[token] = 0
            freqdict[token] += 1
    freqfile = output_dir.parent / "freq.log"
    with open(freqfile, "w") as f:
        freqlist = sorted(freqdict.items(), key=lambda x: x[1], reverse=True)
        for token, freq in freqlist:
            f.write(f"{token}: {freq}\n")
    tlist = []
    for img_path, tags in tag_results:
        if img_path.stem.endswith("_face"):
            pcaption = "portrait of a"
        else:
            pcaption = caption_prefix
        tokens = tags.split(", ")
        out_tokens = [f"{pcaption} {instance_prompt}"]
        for token in tokens:
            pres = pattern.search(token)
            if pres is not None:
                continue
            # if freqdict[token] > 0.7*len(tag_results):
            #     continue
            out_tokens.append(token)
            if token not in freqdict:
                freqdict[token] = 0
            freqdict[token] += 1
        out_tag = ", ".join(out_tokens)
        tlist.append((img_path, out_tag))

    with open(output_dir / 'metadata.jsonl', 'w') as outfile:
        for img_path, out_tag in tlist:
            shutil.copy(img_path, output_dir / img_path.name)
            entry = {"file_name":img_path.name, "prompt": out_tag}
            json.dump(entry, outfile)
            outfile.write('\n')

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--instance_dir", type=str, 
                        default=None, required=True,
                        help="orign instance images path")
    parser.add_argument("--subfolders", action="store_true",
                        help="process subfolders")
    parser.add_argument("--class_prompt", type=str,
                        default="", required=True,
                        help="regular class name")
    parser.add_argument("--output", type=str,
                        default=None, required=True,
                        help="output path")
    parser.add_argument("--interrogator", type=str,
                        default="wd14",
                        help="interrogator")
    parser.add_argument("--caption_prefix", type=str,
                        default="photo of a",
                        help="caption prefix")
    parser.add_argument("--init_new", action="store_true",
                        default=False,
                        help="initialize new training")
    
    return parser.parse_args(input_args)
    
def main(args, wd14args = None):
    instance_dir = Path(args.instance_dir)
    output_dir = Path(args.output)
    class_prompt = args.class_prompt
    caption_prefix = args.caption_prefix
    
    dir_list = []
    if args.subfolders:
        dir_list = list(instance_dir.iterdir())
    else:
        dir_list = [instance_dir]
    dir_list = sorted(dir_list, key=lambda x: x.name)
    
    for tdir in dir_list:
        instance = "<s0><s1>"
        instance_prompt = f"{instance} {class_prompt}"
        imgs_and_paths = read_data(tdir)
        image_dir = output_dir / tdir.name / "images"
        if args.init_new and os.path.exists(output_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        if args.interrogator == "blip":
            blip_prepare_data(imgs_and_paths, image_dir, instance_prompt, caption_prefix)
        elif args.interrogator == "wd14":
            wd14_prepare_data(imgs_and_paths, image_dir, instance_prompt, caption_prefix)
    return 0

if __name__ == "__main__":
    args = parse_args()
    
    if args.interrogator == "blip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16)
        blip_model = blip_model.to(device)
        ret = main(args)
    elif args.interrogator == "wd14":
        wd14args = wd14_tagger.ImageTaggerArgs()
        wd14args.undesired_tags = "1girl,1boy,1women,1man,1person,child,solo"
        wd14_model, wd14_all_tags = wd14_tagger.load_model_and_tags(wd14args)
        ret = main(args, wd14args)
    else:
        raise NotImplementedError("Integrator not implemented")
    sys.exit(ret)