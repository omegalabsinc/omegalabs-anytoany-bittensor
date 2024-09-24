import argparse
import io
from pathlib import Path
import warnings
from pprint import pformat

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from imagebind.models import imagebind_model
    from models.imagebind_wrapper import get_imagebind_v2, V2_PATH
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
    a.add_argument('--key-to-idx-path', type=Path, default='ds/sam_llava/key_to_idx.pt')
    a.add_argument('--sam-sources-path', type=Path, default='ds/sam_llava/sources.csv')
    a.add_argument('--output-dir', type=Path, default='ds/sam_llava/tmp')
    a.add_argument('--progress-period', type=int, default=1024)
    a.add_argument('--write-period', type=int, default=100*1024)
    a.add_argument('--v2', action='store_true', default=True)
    return a.parse_args()


if __name__ == '__main__':
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
        imagebind_model = get_imagebind_v2(path=V2_PATH).imagebind_huge(pretrained=True)
    else:
        print('Initializing and loading Imagebind model...')
        imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model.eval()
    imagebind_model.to(device)

    def imagebind_embed(img_tensor):
        return imagebind_model(
            {ModalityType.VISION: img_tensor.unsqueeze(0).to(device)}
        )[ModalityType.VISION].squeeze(0).cpu()

    print('Loading caption dataset...')
    caption_dataset = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")["train"]
    # caption_dataset = caption_dataset.train_test_split(seed=1337, test_size=0.05)[split_name]

    key_to_caption_idx_path = args.key_to_idx_path
    if key_to_caption_idx_path.exists():
        key_to_caption_idx = torch.load(key_to_caption_idx_path)
    else:
        print('building key->caption-idx...')
        key_to_caption_idx = {}
        for idx, item in enumerate(caption_dataset):
            key_to_caption_idx[item["__key__"]] = idx
        torch.save(key_to_caption_idx, key_to_caption_idx_path)

    print(f'Loading SAM tar archives from {args.sam_sources_path}...')
    sam_dataset = FileOpener(
        [str(args.sam_sources_path)]
    ).parse_csv(
        delimiter=' ',
        skip_lines=1
    ).map(
        lambda t: t[4] # just need url
    ).open_files_by_iopath(mode='rb').load_from_tar()

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
    captions, ib_embeds, clip_embeds = [], [], []

    print('Processing images...')

    for i, (key_path, value) in enumerate(sam_dataset):
        key_path = Path(key_path)
        if key_path.suffix == ".jpg":
            key = key_path.stem
            caption_idx = key_to_caption_idx.get(key)
            if caption_idx:
                captions.append(caption_dataset[caption_idx]["txt"])

                img = Image.open(io.BytesIO(value.read())).convert('RGB')

                with torch.no_grad():
                    # imagebind
                    ib_embeds.append(imagebind_embed(image_transform(img)))

                    # clip
                    img = clip_pipe.feature_extractor(images=img, return_tensors="pt").pixel_values
                    img = img.to(device=device, dtype=dtype)
                    clip_embeds.append(clip_pipe.image_encoder(img).image_embeds.squeeze(0).cpu())

                if len(captions) % args.progress_period == 0:
                    print(len(captions), '...')
                if len(captions) > args.write_period:
                    print(f'Writing archives: {archive_idx}')
                    torch.save(captions, args.output_dir / f'{archive_idx:02d}.caption.pt')
                    torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
                    torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')
                    archive_idx += 1 
                    captions, ib_embeds, clip_embeds = [], [], []

    print(f'Writing archives: {archive_idx}')
    torch.save(captions, args.output_dir / f'{archive_idx:02d}.caption.pt')
    torch.save(torch.stack(ib__embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
    torch.save(torch.stack(clip_embeds), args.output_dir / f'{archive_idx:02d}.clip_embed.pt')

