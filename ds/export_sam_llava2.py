import os
import argparse
import io
from pathlib import Path
import warnings
from pprint import pformat
import tempfile
from contextlib import contextmanager
import tarfile
import shutil
import json
import requests
import urllib.parse
import hashlib

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

class BatchFileProcessor:
    def __init__(self, sam_dataset, temp_dir, checkpoint_file, batch_size=5):
        self.sam_dataset = sam_dataset
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_files = set()
        self.existing_files = self.get_existing_tar_files()
        self.load_checkpoint()

    def get_existing_tar_files(self):
        return [f for f in os.listdir(self.temp_dir) if f.endswith('.tar')]

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.processed_files = set(checkpoint['processed_files'])
                self.last_archive_idx = checkpoint.get('last_archive_idx', -1)
        else:
            self.last_archive_idx = -1

    def save_checkpoint(self, archive_idx):
        print(f'Saving checkpoint with archive index {archive_idx} to {self.checkpoint_file}...')
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_files': list(self.processed_files),
                'last_archive_idx': archive_idx
            }, f)

    def process_batches(self):
        # Process existing files first
        for filename in self.existing_files:
            temp_file = os.path.join(self.temp_dir, filename)
            if temp_file not in self.processed_files:
                yield temp_file
                self.processed_files.add(temp_file)

        batch = []
        for file_uri in self.sam_dataset:
            if file_uri.startswith('http'):
                filename = self.sanitize_filename(file_uri)
            else:
                filename = os.path.basename(file_uri)
            
            temp_file = os.path.join(self.temp_dir, filename)
            
            if temp_file in self.processed_files:
                continue  # Skip already processed files
            
            if os.path.exists(temp_file):
                # Prioritize existing files
                yield temp_file
                self.processed_files.add(temp_file)
            else:
                batch.append(file_uri)
            
            if len(batch) >= self.batch_size:
                yield from self.process_batch(batch)
                batch = []
        
        if batch:
            yield from self.process_batch(batch)

    def process_batch(self, batch):
        for file_uri in batch:
            if file_uri.startswith('http'):
                filename = self.sanitize_filename(file_uri)
                temp_file = os.path.join(self.temp_dir, filename)
                try:
                    self.download_file(file_uri, temp_file)
                    yield temp_file
                    self.processed_files.add(temp_file)
                except Exception as e:
                    print(f"Error downloading file {file_uri}: {str(e)}")
            else:
                temp_file = os.path.join(self.temp_dir, os.path.basename(file_uri))
                shutil.copy(file_uri, temp_file)
                yield temp_file
                self.processed_files.add(temp_file)

    def cleanup_batch(self):
        for file in self.processed_files:
            try:
                os.remove(file)
                print(f"Removed processed file: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {str(e)}")
        self.processed_files.clear()

    def sanitize_filename(self, url):
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        filename = filename.split('?')[0]
        if not filename or '.' not in filename:
            filename = hashlib.md5(url.encode()).hexdigest() + '.tar'
        return filename

    def download_file(self, url, temp_file):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def process_tar_file(file_path, key_to_caption_idx, caption_dataset, image_transform, imagebind_embed):
    captions, ib_embeds = [], []
    with tarfile.open(file_path, 'r:*') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.jpg'):
                f = tar.extractfile(member)
                if f is not None:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
                    key = Path(member.name).stem
                    caption_idx = key_to_caption_idx.get(key)
                    if caption_idx:
                        captions.append(caption_dataset[caption_idx]["txt"])
                        with torch.no_grad():
                            ib_embeds.append(imagebind_embed(image_transform(img)))
    return captions, ib_embeds

def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--key-to-idx-path', type=Path, default='ds/sam_llava/key_to_idx.pt')
    a.add_argument('--sam-sources-path', type=Path, default='ds/sam_llava/sources.csv')
    a.add_argument('--output-dir', type=Path, default='ds/sam_llava/tmp')
    a.add_argument('--progress-period', type=int, default=1024)
    a.add_argument('--write-period', type=int, default=100*1024)
    a.add_argument('--v2', action='store_true', default=True)
    a.add_argument('--temp-dir', type=Path, default='ds/sam_llava/tmpfiles', help='Temporary directory for downloaded files')
    return a.parse_args()

def main(args):
    print(f'args:\n{pformat(vars(args))}')

    # Set up temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()

    device = torch.device('cuda:0')
    dtype = torch.float16

    print('Loading clip pipeline...')
    clip_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=dtype)
    clip_pipe.to(device)

    # load imagebind
    if args.v2:
        print('Initializing and loading Imagebind v2 model...')
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

    print('Loading caption dataset...')
    caption_dataset = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")["train"]

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
        delimiter='\t',
        skip_lines=1
    ).map(
        lambda t: t[1] if len(t) > 1 else None
    ).filter(
        lambda x: x is not None
    )

    # Add a check to see if the dataset is empty
    sample_data = next(iter(sam_dataset), None)
    if sample_data is None:
        print("Warning: The dataset is empty. Please check your input file.")
        exit()

    checkpoint_file = args.output_dir / 'sam_llava_checkpoint.json'
    batch_processor = BatchFileProcessor(sam_dataset, args.temp_dir, checkpoint_file, batch_size=5)
    print(f'Found {len(batch_processor.existing_files)} existing .tar files in the temporary directory.')

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

    archive_idx = batch_processor.last_archive_idx + 1
    captions, ib_embeds = [], []

    print(f'Processing images starting from archive index {archive_idx}...')

    for batch_idx, tar_file in enumerate(batch_processor.process_batches()):
        print(f"Processing file {batch_idx + 1}: {tar_file}")
        batch_captions, batch_ib_embeds = process_tar_file(
            tar_file, key_to_caption_idx, caption_dataset, image_transform, imagebind_embed
        )
        captions.extend(batch_captions)
        ib_embeds.extend(batch_ib_embeds)

        print(f"Current number of processed captions: {len(captions)}")
        print(f"Current number of processed embeddings: {len(ib_embeds)}")

        # Write out files more frequently
        if len(captions) >= min(args.write_period, 1000):  # Write at least every 1000 samples
            print(f'Writing archives: {archive_idx}')
            torch.save(captions, args.output_dir / f'{archive_idx:02d}.caption.pt')
            torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
            batch_processor.save_checkpoint(archive_idx)
            print(f"Checkpoint saved for archive index: {archive_idx}")
            archive_idx += 1
            captions, ib_embeds = [], []

        if (batch_idx + 1) % 5 == 0:
            print(f"Cleaning up after processing {batch_idx + 1} files")
            batch_processor.cleanup_batch()

    # Write any remaining data
    if captions:
        print(f'Writing final archives: {archive_idx}')
        torch.save(captions, args.output_dir / f'{archive_idx:02d}.caption.pt')
        torch.save(torch.stack(ib_embeds), args.output_dir / f'{archive_idx:02d}.ib_embed.pt')
        batch_processor.save_checkpoint(archive_idx)
        print(f"Final checkpoint saved for archive index: {archive_idx}")

    batch_processor.cleanup_batch()  # Clean up any remaining files
    print("Processing completed.")

if __name__ == '__main__':
    args = parse_args()
    main(args)