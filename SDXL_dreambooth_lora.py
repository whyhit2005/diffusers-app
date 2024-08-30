# %% [markdown]
# ## Fine-tuning Stable Diffusion XL with DreamBooth and LoRA on a free-tier Colab Notebook 🧨
# 
# In this notebook, we show how to fine-tune [Stable Diffusion XL (SDXL)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl) with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth) and [LoRA](https://huggingface.co/docs/diffusers/main/en/training/lora) on a T4 GPU.
# 
# SDXL consists of a much larger UNet and two text encoders that make the cross-attention context quite larger than the previous variants.
# 
# So, to pull this off, we will make use of several tricks such as gradient checkpointing, mixed-precision, and 8-bit Adam. So, hang tight and let's get started 🧪

# %% [markdown]
# ## Setup 🪓

# %% [markdown]
# ## Dataset 🐶

# %% [markdown]
# **Let's get our training data!**
# For this example, we'll download some images from the hub
# 
# If you already have a dataset on the hub you wish to use, you can skip this part and go straight to: "Prep for
# training 💻" section, where you'll simply specify the dataset name.
# 
# If your images are saved locally, and/or you want to add BLIP generated captions,
# pick option 1 or 2 below.
# 
# 

# %% [markdown]
# **Option 2:** download example images from the hub:

# %% [markdown]
# Preview the images:

# %%
from PIL import Image

def image_grid(imgs, rows, cols, resize=256):

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# %%
import glob
# change path to display images from your local dir
local_dir = "/home/wangyh/images/wangxiaojiu/1024/"
img_paths = local_dir+"*.png"

imgs = [Image.open(path) for path in glob.glob(img_paths)]
print(len(imgs))
num_imgs_to_preview = 5
image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# %%
reg_dir = "/home/wangyh/images/cls-crop/"
reg_paths = reg_dir+"*.png"
reg_imgs = [Image.open(path) for path in glob.glob(reg_paths)]
print(len(reg_imgs))
num_imgs_to_preview = 5
image_grid(reg_imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# %% [markdown]
# ### Generate custom captions with BLIP
# Load BLIP to auto caption your images:

# %%
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor and the captioning model
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

# captioning utility
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# %% [markdown]
# Now let's add the concept token identifier (e.g. TOK) to each caption using a caption prefix.
# Feel free to change the prefix according to the concept you're training on!
# - for this example we can use "a photo of TOK," other options include:
#     - For styles - "In the style of TOK"
#     - For faces - "photo of a TOK person"
# - You can add additional identifiers to the prefix that can help steer the model in the right direction.
# -- e.g. for this example, instead of "a photo of TOK" we can use "a photo of TOK dog" / "a photo of TOK corgi dog"

# %%
import os
import shutil
from tqdm import tqdm

output_dir = "./output/wxj-db-lora/"
if True and os.path.exists(output_dir):
    shutil.rmtree(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_dir = os.path.join(output_dir, "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


# %%
import glob
from PIL import Image

# create a list of (Pil.Image, path) pairs
local_dir = "/home/wangyh/images/wangxiaojiu/1024/"
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.png")]
local_dir = "/home/wangyh/images/wangxiaojiu/crop/"
imgs_and_paths += [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.png")]
reg_dir = "/home/wangyh/images/cls-crop/"
reg_imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{reg_dir}*.png")]

# %%
import json
from tqdm import tqdm
import shutil
import random
import hashlib
    
caption_prefix = "photo of a ohwx child, " #@param
reg_prefix = "photo of a child, "
with open(f'{image_dir}/metadata.jsonl', 'w') as outfile:
    for img in tqdm(imgs_and_paths):
        caption = caption_prefix + caption_images(img[1]).split("\n")[0]
        basename = hashlib.sha1(img[1].tobytes()).hexdigest()+".png"
        tarpath = os.path.join(image_dir, basename)
        shutil.copy(img[0], tarpath)
        entry = {"file_name":basename, "prompt": caption}
        json.dump(entry, outfile)
        outfile.write('\n')
    # rindexs = [random.randint(0, len(reg_imgs_and_paths)) for _ in range(200)]
    # for img in tqdm(reg_imgs_and_paths[:200]):
    #     caption = reg_prefix + caption_images(img[1]).split("\n")[0]
    #     basename = hashlib.sha1(img[1].tobytes()).hexdigest()+".png"
    #     tarpath = os.path.join(image_dir, basename)
    #     shutil.copy(img[0], tarpath)
    #     entry = {"file_name":basename, "prompt": caption}
    #     json.dump(entry, outfile)
    #     outfile.write('\n')

# %% [markdown]
# Free some memory:

# %%
import gc

# delete the BLIP pipelines and free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## Prep for training 💻

# %% [markdown]
# Initialize `accelerate`:

# %% [markdown]
# ## Train! 🔬

# %% [markdown]
# #### Set Hyperparameters ⚡
# To ensure we can DreamBooth with LoRA on a heavy pipeline like Stable Diffusion XL, we're using:
# 
# * Gradient checkpointing (`--gradient_accumulation_steps`)
# * 8-bit Adam (`--use_8bit_adam`)
# * Mixed-precision training (`--mixed-precision="fp16"`)

# %% [markdown]
# ### Launch training 🚀🚀🚀

# %% [markdown]
# To allow for custom captions we need to install the `datasets` library, you can skip that if you want to train solely
#  with `--instance_prompt`.
# In that case, specify `--instance_data_dir` instead of `--dataset_name`

# %% [markdown]
#  - Use `--output_dir` to specify your LoRA model repository name!
#  - Use `--caption_column` to specify name of the cpation column in your dataset. In this example we used "prompt" to
#  save our captions in the
#  metadata file, change this according to your needs.

# %%
#!/usr/bin/env bash
!accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="output/wxj-db-lora/images" \
  --output_dir="output/wxj-db-lora/models" \
  --caption_column="prompt"\
  --mixed_precision="bf16" \
  --instance_prompt="photo of a ohwx child" \
  --resolution=1024 \
  --train_batch_size=10 \
  --repeats=10 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=4e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --max_train_steps=1000 \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=10 \
  --seed=43 \
  --rank=128 \
  --allow_tf32 \
  --report_to=wandb

# %% [markdown]
# Let's generate some images with it!

# %% [markdown]
# ## Inference 🐕

# %%
import torch
import os
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
output_dir = "./output/wxj-db-lora/"
lora_w = os.path.join(output_dir, "models", "pytorch_lora_weights.safetensors")
pipe.load_lora_weights(lora_w)
_ = pipe.to("cuda")

# %%
import hashlib
prompt = "a ohwx child, a dog, on the beach, night, stars background, Disney style" # @param
prompt = "glamor photo of a ohwx child with short curly black hair, playing a musical instrument, smiling excitedly, Upper body from behind, on a rocky edge of a cliff overlooking a misty forested valley at dawn, Cinematic lighting with moody ambiance, Dutch angle, Lumix GH5, in style of by Steve McCurry"

num_sample = 20
sample_dir = os.path.join(output_dir, "samples")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
generator = torch.Generator("cuda").manual_seed(4572436)
pipe.disable_attention_slicing()
pip.enable_xformers_memory_efficient_attention()
images = pipe(prompt=prompt, num_inference_steps=50, num_images_per_prompt = num_sample,
                height=1024, width=1024, generator=generator).images
for image in images:
    fname = f"{i:03d}"
    image.save(os.path.join(sample_dir, fname+".png"), format="png")
col = 5
row = num_sample // col
image_grid(images[:col*row], row, col)

# %%
import gc

# delete the BLIP pipelines and free up some memory
del vae, pipe
gc.collect()
torch.cuda.empty_cache()

# %%



