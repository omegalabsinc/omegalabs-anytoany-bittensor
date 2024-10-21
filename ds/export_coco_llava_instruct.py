import argparse
from pathlib import Path
from pprint import pformat
import warnings
import json

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from imagebind.models import imagebind_model
    from models.imagebind_wrapper import get_imagebind_v2, V2_PATH
    from imagebind.models.imagebind_model import ModalityType
    from imagebind.models.multimodal_preprocessors import SimpleTokenizer

import torch
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

from diffusers import DiffusionPipeline

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--output-dir', type=Path, default='ds/coco_llava_instruct/tmp')
    a.add_argument('--progress-period', type=int, default=1024)
    a.add_argument('--write-period', type=int, default=100*1024)
    a.add_argument('--v2', action='store_true', default=True)
    return a.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f'args:\n{pformat(vars(args))}')

    device = torch.device('cuda:0')
    dtype = torch.float16

    print('Loading clip pipeline...')
    clip_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=dtype)
    clip_pipe.to(device)

    # load imagebind
    if args.v2:
        print('Initializing and loading Imagebind v2 model...')
        #imagebind_model = get_imagebind_v2(path=V2_PATH).imagebind_huge(pretrained=True)
        imagebind_model = get_imagebind_v2(path=V2_PATH)
    else:
        print('Initializing and loading Imagebind model...')
        imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model.eval()
    imagebind_model.to(device)

    def imagebind_embed(img_tensor):
        return imagebind_model(
            {ModalityType.VISION: img_tensor.unsqueeze(0).to(device)}
        )[ModalityType.VISION].squeeze(0).cpu()

    ds_path = Path('ds/coco_llava_instruct')
    coco_path = ds_path / 'train2014'

    print('Loading conversation dataset...')
    with open(ds_path / 'llava_instruct_150k.json') as fin:
        conversation_dataset = json.load(fin)
    # conversation_dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")["train"]

    image_transform = transforms.Compose([
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    archive_idx = 0
    conversations, ib_embeds, clip_embeds = [], [], []

    print('Processing images...')

    for item in conversation_dataset:
        img_path = coco_path / ('COCO_train2014_' + item['image'])
        if not img_path.exists():
            continue

        conversations.append(item['conversations']) # each is really a single conversation

        img = Image.open(img_path).convert('RGB')

        with torch.no_grad():
            # imagebind
            ib_embeds.append(imagebind_embed(image_transform(img)))

            # clip
            #img = clip_pipe.feature_extractor(images=img, return_tensors="pt").pixel_values
            #img = img.to(device=device, dtype=dtype)
            #clip_embeds.append(clip_pipe.image_encoder(img).image_embeds.squeeze(0).cpu())

        if len(conversations) % args.progress_period == 0:
            print(len(conversations), '...')
        if len(conversations) > args.write_period:
            print(f'Writing archives: {archive_idx}')
            torch.save(conversations, args.output_dir / f'{archive_idx:02d}.caption.pt')
            torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
            #torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')
            archive_idx += 1 
            conversations, ib_embeds, clip_embeds = [], [], []

    print(f'Writing archives: {archive_idx}')
    torch.save(conversations, args.output_dir / f'{archive_idx:02d}.caption.pt')
    torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
    #torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')


