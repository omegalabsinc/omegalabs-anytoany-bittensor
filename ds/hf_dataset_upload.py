import os
import json
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

PROGRESS_FILE = 'progress.json'
OUTPUT_PARQUET = 'output.parquet'
HF_OUTPUT_PARQUET = 'output.parquet'

def truncate_list(lst, max_items=5):
    if len(lst) <= max_items:
        return str(lst)
    return f"[{', '.join(map(str, lst[:max_items]))}, ... ({len(lst)} items)]"

def analyze_dataset(dataset_name, num_samples=10):
    print(f"Analyzing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    columns = dataset.column_names
    print(f"Columns: {columns}")

    print(f"\nAnalyzing the first {num_samples} samples:")
    for i, sample in enumerate(dataset.select(range(num_samples))):
        print(f"\nSample {i}:")
        for column in columns:
            value = sample[column]
            if isinstance(value, (list, tuple)):
                print(f"  {column}: {type(value).__name__} of length {len(value)}")
                if column == 'caption':
                    print(f"    Value: {truncate_list(value)}")
            else:
                print(f"  {column}: {type(value).__name__}")
                if column == 'caption':
                    print(f"    Value: {value}")

    print("\nDataset statistics:")
    print(f"Total number of samples: {len(dataset)}")
    for column in columns:
        feature = dataset.features[column]
        if isinstance(feature, pa.lib.StringType):
            print(f"  {column}: text column")
        elif isinstance(feature, pa.lib.ListType):
            print(f"  {column}: list of {feature.value_type}, length {len(dataset[0][column])}")
        else:
            print(f"  {column}: {feature}")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed_files': [], 'total_samples': 0, 'uploaded': False}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def truncate_list(lst, max_items=5):
    if len(lst) <= max_items:
        return str(lst)
    return f"[{', '.join(map(str, lst[:max_items]))}, ... ({len(lst)} items)]"

def process_pt_file(file_path):
    try:
        print(f"Processing file: {file_path}")
        data = torch.load(file_path)
        print(f"Loaded data type: {type(data)}")
        
        if isinstance(data, list):  # captions
            print(f"Found list data of length {len(data)}")
            results = []
            for item in data:
                #print(f"  Item type: {type(item)}")
                #print(f"  Item: {item}")
                #os._exit(1)
                if isinstance(item, list) and all(isinstance(d, dict) for d in item):
                    caption = [{'from': d['from'], 'value': d['value']} for d in item if 'from' in d and 'value' in d]
                    if caption:
                        results.append({
                            'caption': caption,
                            'ib_embed': None  # Initialize with None
                        })
                elif isinstance(item, str):
                    results.append({
                        'caption': item,
                        'ib_embed': None  # Initialize with None
                    })
            print(f"Processed {len(results)} valid captions")
            return results
        elif isinstance(data, torch.Tensor) and data.dim() == 2:  # embeddings
            print(f"Found tensor data of shape {data.shape}")
            if data.shape[1] == 1024:  # ImageBind embeddings
                results = [{'caption': None, 'ib_embed': emb.tolist()} for emb in data]
                print(f"Processed {len(results)} ImageBind embeddings")
                return results
            else:
                print(f"Unexpected tensor shape: {data.shape}")
        else:
            print(f"Unexpected data type or structure: {type(data)}")
        
        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def merge_samples(samples):
    merged = {
        'caption': None,
        'ib_embed': None
    }
    for sample in samples:
        if sample['caption'] is not None:
            merged['caption'] = sample['caption']
        if sample['ib_embed'] is not None:
            merged['ib_embed'] = sample['ib_embed']
    return merged

def create_parquet_files(input_dir, output_dir, files_per_group=50):
    output_path = os.path.join(output_dir, OUTPUT_PARQUET)
    writer = None

    print(f"Will attempt to create file at: {output_path}")
    print(f"Input directory: {input_dir}")

    #schema = pa.schema([
        #('caption', pa.list_(pa.struct([('from', pa.string()), ('value', pa.string())]))),
        #('ib_embed', pa.list_(pa.float64(), 1024))
    #])
    # sam_llava schema
    schema = pa.schema([
        ('caption', pa.string()),
        ('ib_embed', pa.list_(pa.float64(), 1024))
    ])

    try:
        # Get all PT files and sort them
        pt_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
        
        # Group files by their index
        file_groups = {}
        for file in pt_files:
            # Extract the index from the filename (e.g., "00" from "00.caption.pt")
            index = int(file.split('.')[0])
            group_num = index // files_per_group
            if group_num not in file_groups:
                file_groups[group_num] = []
            file_groups[group_num].append(file)

        total_samples = 0
        
        # Process each group
        for group_num in sorted(file_groups.keys()):
            output_path = os.path.join(output_dir, f'output_{group_num:03d}.parquet')
            print(f"\nProcessing group {group_num} -> {output_path}")
            
            writer = pq.ParquetWriter(output_path, schema)
            group_samples = 0

            # Sort files within the group to ensure captions and embeddings align
            group_files = sorted(file_groups[group_num])
            
            # Process files in pairs (caption and embedding)
            caption_files = [f for f in group_files if 'caption' in f]
            embed_files = [f for f in group_files if 'ib_embed' in f]

            print(f"Found {len(caption_files)} caption files and {len(embed_files)} embedding files in group {group_num}")

            for caption_file, embed_file in zip(caption_files, embed_files):
                caption_path = os.path.join(input_dir, caption_file)
                embed_path = os.path.join(input_dir, embed_file)
                
                print(f"Processing pair: {caption_file} and {embed_file}")
                
                captions = torch.load(caption_path)
                embeddings = torch.load(embed_path)

                # Create and write samples
                for caption, embedding in zip(captions, embeddings):
                    merged_sample = {
                        'caption': caption,
                        'ib_embed': embedding.tolist()
                    }
                    
                    table = pa.Table.from_pylist([merged_sample], schema=schema)
                    writer.write_table(table)
                    group_samples += 1

            writer.close()
            total_samples += group_samples
            print(f"Completed group {group_num} with {group_samples} samples")

        print(f"\nProcessing completed:")
        print(f"Total samples processed: {total_samples}")
        print(f"Total parquet files created: {len(file_groups)}")

        return total_samples, len(file_groups)

    except Exception as e:
        print(f"An error occurred while creating the parquet files: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def load_upload_progress():
    progress_file = 'upload_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'uploaded_files': []}

def save_upload_progress(uploaded_files):
    progress_file = 'upload_progress.json'
    with open(progress_file, 'w') as f:
        json.dump({'uploaded_files': uploaded_files}, f)

def upload_to_huggingface(output_dir, dataset_name):
    # Load upload progress
    progress = load_upload_progress()
    uploaded_files = set(progress['uploaded_files'])

    # Find all parquet files
    parquet_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.parquet')])
    
    if not parquet_files:
        print("No parquet files found in the output directory.")
        return

    print(f"Found {len(parquet_files)} parquet files.")
    print(f"Previously uploaded: {len(uploaded_files)} files.")

    #api = HfApi(token="XXXXXXXX")
    # using HF_API_KEY env variable should work
    api = HfApi()

    # Upload first file with dataset creation
    """
    if not uploaded_files and parquet_files:
        first_file = parquet_files[0]
        first_file_path = os.path.join(output_dir, first_file)
        print(f"Creating dataset with first file: {first_file}")
        try:
            dataset = Dataset.from_parquet(first_file_path)
            dataset.push_to_hub(dataset_name, private=True)
            uploaded_files.add(first_file)
            save_upload_progress(list(uploaded_files))
            print(f"Successfully uploaded initial file: {first_file}")
        except Exception as e:
            print(f"Error uploading initial file {first_file}: {str(e)}")
            return
    """

    # Upload remaining files
    for parquet_file in parquet_files:
        if parquet_file in uploaded_files:
            print(f"Skipping already uploaded file: {parquet_file}")
            continue

        file_path = os.path.join(output_dir, parquet_file)
        print(f"Uploading {parquet_file} to {dataset_name}...")

        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=parquet_file,  # Use the same filename in the repo
                repo_id=dataset_name,
                repo_type="dataset"
            )
            uploaded_files.add(parquet_file)
            save_upload_progress(list(uploaded_files))
            print(f"Successfully uploaded: {parquet_file}")
        except Exception as e:
            print(f"Error uploading {parquet_file}: {str(e)}")
            print("Will try to continue with remaining files...")
            continue

    total_uploaded = len(uploaded_files)
    print(f"\nUpload process completed:")
    print(f"Total files uploaded: {total_uploaded}/{len(parquet_files)}")
    
    if total_uploaded == len(parquet_files):
        print("All files successfully uploaded!")
    else:
        print(f"Some files were not uploaded. Check the logs and try running the upload again.")

def main():
    #print("Analyzing existing dataset")
    #analyze_dataset("nimapourjafar/sam_llava")

    #input_dir = 'ds/vision_flan'
    #output_dir = 'ds/vision_flan'
    #dataset_name = 'xzistance/vision_flan'

    #input_dir = 'ds/coco_llava_instruct/tmp'
    #output_dir = 'ds/coco_llava_instruct'
    #dataset_name = 'xzistance/coco_llava_instruct'

    input_dir = 'ds/sam_llava/tmp'
    output_dir = 'ds/sam_llava'
    dataset_name = 'xzistance/sam_llava'

    os.makedirs(output_dir, exist_ok=True)
        
    #print("Step 1: Creating parquet file(s)")
    #total_samples, total_files = create_parquet_files(input_dir, output_dir)
    #print(f"Created {total_files} parquet files with {total_samples} total samples")

    print("Step 2: Uploading to Hugging Face")
    upload_to_huggingface(output_dir, dataset_name)

if __name__ == '__main__':
    main()