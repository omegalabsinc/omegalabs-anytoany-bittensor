import os
import time
import traceback
import tempfile
import shutil
import random
from typing import Dict, Any
import bittensor as bt
from datasets import load_dataset, Audio

from neurons.miner_model_assistant import MinerModelAssistant

def run_voice_mos_evaluation(
    container_url: str,
) -> Dict[str, Any]:
    """
    Run voice quality evaluation using v2v endpoint and MOS scoring.
    """
    temp_audio_dir = None
    try:
        # Create temp directory for audio files
        temp_audio_dir = tempfile.mkdtemp(prefix="voice_mos_")
        bt.logging.info(f"Created temp audio directory: {temp_audio_dir}")

        # Inference phase - call v2v endpoint, save audio files
        inference_results = inference_voice_mos(
            container_url=container_url,
            temp_audio_dir=temp_audio_dir
        )

        # Scoring phase - get MOS score from saved audio
        mos_score = get_mos_score(temp_audio_dir)

        return {
            'voice_mos_score': mos_score,
            'voice_mos_details': {
                'samples_processed': inference_results['total_samples'],
                'datasets_evaluated': inference_results['datasets'],
                'inference_success_rate': inference_results['success_rate'],
                'sample_details': inference_results['sample_details']
            }
        }

    except Exception as e:
        bt.logging.error(f"Voice MOS evaluation failed: {e}")
        bt.logging.error(traceback.format_exc())
        return {
            'voice_mos_score': 0.0,
            'voice_mos_details': {
                'error': str(e),
                'samples_processed': 0,
                'datasets_evaluated': [],
                'inference_success_rate': 0.0,
                'sample_details': []
            }
        }

    finally:
        # Ensure cleanup even on errors
        if temp_audio_dir and os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)
            bt.logging.info(f"Cleaned up temp audio directory: {temp_audio_dir}")


def inference_voice_mos(
    container_url: str,
    temp_audio_dir: str,
) -> Dict[str, Any]:
    """
    Call v2v endpoint on VoiceBench samples and save audio outputs.
    Using only 2 random samples per dataset.
    """
    # VoiceBench datasets (same as used in voicebench_adapter.py)
    VOICEBENCH_DATASETS = {
        'commoneval': ['test'],
        'wildvoice': ['test'],
        'ifeval': ['test'],
        'advbench': ['test']
    }
    miner_assistant = MinerModelAssistant(base_url=container_url)

    total_samples = 0
    success_count = 0
    sample_details = []
    datasets_evaluated = []
    # how many samples to use here?? I think 2 random from each dataset should be fine for now.

    for dataset_name, splits in VOICEBENCH_DATASETS.items():
        bt.logging.info(f"Processing dataset: {dataset_name}")
        datasets_evaluated.append(dataset_name)

        for split in splits:
            # Load dataset
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            # Get random samples
            dataset_size = len(dataset)
            sample_count = min(2, dataset_size)
            sample_indices = random.sample(range(dataset_size), sample_count)

            for sample_idx in sample_indices:
                try:
                    total_samples += 1
                    sample = dataset[sample_idx]

                    # Get audio data
                    audio_data = sample['audio']['array']
                    sample_rate = sample['audio']['sampling_rate']

                    # Call v2v endpoint
                    start_time = time.time()
                    response = miner_assistant.inference_v2v(
                        audio_array=audio_data,
                        sample_rate=sample_rate
                    )
                    inference_time = time.time() - start_time

                    # Save audio file with flat naming
                    filename = f"{dataset_name}_{split}_{sample_idx:04d}.wav"
                    audio_path = os.path.join(temp_audio_dir, filename)

                    # Save WAV bytes directly to file
                    with open(audio_path, 'wb') as f:
                        f.write(response['audio_wav_bytes'])

                    success_count += 1
                    bt.logging.debug(f"Saved audio: {filename}")

                    sample_details.append({
                        'dataset': dataset_name,
                        'split': split,
                        'sample_index': sample_idx,
                        'filename': filename,
                        'inference_time': inference_time,
                        'success': True
                    })
                    
                except Exception as e:
                    bt.logging.error(f"Error processing {dataset_name}_{sample_idx}: {e}")
                    traceback.print_exc()
                    inference_time = time.time() - start_time if 'start_time' in locals() else 0.0
                    sample_details.append({
                        'dataset': dataset_name,
                        'split': split,
                        'sample_index': sample_idx,
                        'filename': None,
                        'inference_time': inference_time,
                        'success': False,
                        'error': str(e)
                    })

    success_rate = success_count / total_samples if total_samples > 0 else 0.0
    bt.logging.info(f"Voice MOS inference completed: {success_count}/{total_samples} samples successful ({success_rate:.2%})")

    return {
        'total_samples': total_samples,
        'success_count': success_count,
        'success_rate': success_rate,
        'datasets': datasets_evaluated,
        'sample_details': sample_details
    }


def get_mos_score(temp_dir_path: str) -> float:
    """
    Calculate MOS score using UTMOS-v2 model.

    Args:
        temp_dir_path: Path to directory containing audio files

    Returns:
        Average MOS score across all audio files
    """
    import utmosv2

    # List audio files
    audio_files = [f for f in os.listdir(temp_dir_path) if f.endswith('.wav')]
    bt.logging.info(f"Found {len(audio_files)} audio files for MOS scoring")

    if len(audio_files) == 0:
        bt.logging.warning("No audio files found for MOS scoring")
        return 0.0

    # Load model (fresh each time to free GPU for other tasks)
    start_time = time.time()
    bt.logging.info("Loading UTMOS-v2 model...")
    model = utmosv2.create_model(pretrained=True)
    model_load_time = time.time() - start_time
    bt.logging.info(f"UTMOS-v2 model loaded in {model_load_time:.2f} seconds")

    # Run MOS scoring
    start_time = time.time()
    bt.logging.info(f"Running MOS scoring on {len(audio_files)} files...")
    mos_results = model.predict(input_dir=temp_dir_path)
    scoring_time = time.time() - start_time
    bt.logging.info(f"MOS scoring completed in {scoring_time:.2f} seconds")

    # Calculate average MOS score
    mos_scores = [result['predicted_mos'] for result in mos_results]
    average_mos = sum(mos_scores) / len(mos_scores)

    bt.logging.info(f"MOS scoring results:")
    bt.logging.info(f"  Files processed: {len(mos_results)}")
    bt.logging.info(f"  Average MOS: {average_mos:.3f}")
    bt.logging.info(f"  MOS range: {min(mos_scores):.3f} - {max(mos_scores):.3f}")
    bt.logging.info(f"  Total time: {model_load_time + scoring_time:.2f}s (load: {model_load_time:.2f}s, scoring: {scoring_time:.2f}s)")

    # Log individual file scores for debugging
    for result in mos_results[:3]:  # Log first 3 files
        bt.logging.debug(f"  {result['file_path']}: {result['predicted_mos']:.3f}")

    # scale MOS to 0-1 range
    average_mos = average_mos/5.0
    bt.logging.info(f"Scaled Average MOS (0-1): {average_mos:.3f}")
    return average_mos

