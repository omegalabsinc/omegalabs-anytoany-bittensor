import argparse
import io
from pathlib import Path
import warnings
from pprint import pformat
import json

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind.models.multimodal_preprocessors import SimpleTokenizer

import torch
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

from diffusers import DiffusionPipeline

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--output-dir', type=Path, default='ds/vision_flan')
    a.add_argument('--progress-period', type=int, default=1024)
    a.add_argument('--write-period', type=int, default=100*1024)
    a.add_argument('--batch-dim', type=int, default=8)
    return a.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'args:\n{pformat(vars(args))}')

    device = torch.device('cuda:0')
    dtype = torch.float16

    print('Loading clip pipeline...')
    clip_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=dtype)
    clip_pipe.to(device)

    print('Loading imagebind model...')
    imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model.eval()
    imagebind_model.to(device)

    def imagebind_embed(img_tensor):
        return imagebind_model(
            {ModalityType.VISION: img_tensor.to(device)}
        )[ModalityType.VISION].cpu()

    ds_path = Path('ds/vision_flan')
    flan_path = ds_path / 'images_191task_1k'
    with open(ds_path / 'annotation_191-task_1k.json') as fin:
        conversation_dataset = json.load(fin)

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
    conv_batch, ib_batch, clip_batch = [], [], []

    for item in conversation_dataset:
        img_path = flan_path / (item['image'])
        if not img_path.exists():
            continue

        conv_batch.append(item['conversations']) # each is really a single conversation

        img = Image.open(img_path).convert('RGB')

        ib_batch.append(image_transform(img))
        clip_batch.append(img)

        if len(ib_batch) == args.batch_dim:
            conversations.extend(conv_batch)
            with torch.no_grad():
                # imagebind
                ib_embeds.extend(imagebind_embed(torch.stack(ib_batch)))

                # clip
                img_features = clip_pipe.feature_extractor(images=clip_batch, return_tensors="pt").pixel_values
                clip_embeds.extend(
                    clip_pipe.image_encoder(
                        img_features.to(device=device, dtype=dtype)
                    ).image_embeds.cpu()
                )

            conv_batch, ib_batch, clip_batch = [], [], []

        if len(conversations) % args.progress_period == 0:
            print(len(conversations), '...')
        if len(conversations) > args.write_period:
            print(f'Writing archives: {archive_idx}')
            torch.save(conversations, args.output_dir / f'{archive_idx:02d}.caption.pt')
            torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
            torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')
            archive_idx += 1 
            conversations, ib_embeds, clip_embeds = [], [], []

    print(f'Writing archives: {archive_idx}')
    torch.save(conversations, args.output_dir / f'{archive_idx:02d}.caption.pt')
    torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
    torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')


