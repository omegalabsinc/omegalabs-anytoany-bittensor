"""
LLM Judge for VoiceBench Evaluation

This module provides LLM-based evaluation of model responses using OpenAI API
with configuration from vali.env file.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
import bittensor as bt
from dotenv import load_dotenv

# Load environment variables from vali.env
vali_env_path = Path(__file__).parent.parent / "vali.env"
if vali_env_path.exists():
    load_dotenv(vali_env_path)

# Initialize OpenAI client with environment configuration
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)
MODEL = os.getenv("MODEL", "gpt-4o-mini")

print(f"Using LLM model: {MODEL}")

# Evaluation prompts from VoiceBench
META_PROMPT_OPEN = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user's instruction and models' response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don't need to provide any explanations.
"""

META_PROMPT_QA = """
### Question
{prompt}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer? 
Please only output a single "Yes" or "No". Do not output anything else.
""".strip()


def generate_llm_score(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate LLM score for a single evaluation item.
    
    Args:
        item: Dictionary containing 'prompt', 'response', and optionally 'reference'
        
    Returns:
        Item with added 'llm_score' field
    """
    try:
        # Choose prompt based on whether reference answer exists
        if "reference" in item and item["reference"]:
            prompt = META_PROMPT_QA.format(
                prompt=item['prompt'], 
                reference=item['reference'], 
                response=item['response']
            )
        else:
            prompt = META_PROMPT_OPEN.format(
                prompt=item['prompt'], 
                response=item['response']
            )
        
        bt.logging.debug(f"Sending LLM evaluation request for prompt: {item['prompt'][:50]}...")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who tries to help answer the user's question."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=10,  # We only need a short response (score or Yes/No)
            temperature=0.1,  # Low temperature for consistent scoring
            n=1
        )
        
        score_text = response.choices[0].message.content.strip()
        
        # Parse score
        if "reference" in item and item["reference"]:
            # For QA tasks, convert Yes/No to binary score
            llm_score = 1.0 if score_text.lower().startswith('yes') else 0.0
        else:
            # For open-ended tasks, parse numeric score
            try:
                llm_score = float(score_text) / 5.0  # Normalize to 0-1
            except ValueError:
                bt.logging.warning(f"Could not parse LLM score: {score_text}")
                llm_score = 0.0
        
        item['llm_score'] = llm_score
        item['llm_raw_response'] = score_text
        
        bt.logging.debug(f"LLM score: {llm_score} (raw: {score_text})")
        
    except Exception as e:
        bt.logging.error(f"Error getting LLM score: {e}")
        item['llm_score'] = 0.0
        item['llm_error'] = str(e)
    
    return item


def evaluate_responses_with_llm(responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate a list of responses using LLM judge.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        List of responses with LLM scores added
    """
    bt.logging.info(f"Starting LLM evaluation of {len(responses)} responses using {MODEL}")
    
    evaluated_responses = []
    
    for i, response in enumerate(responses):
        bt.logging.debug(f"Evaluating response {i+1}/{len(responses)}")
        
        # Skip if there's no valid response text
        if not response.get('response', '').strip():
            response['llm_score'] = 0.0
            response['llm_error'] = 'Empty response'
            evaluated_responses.append(response)
            continue
            
        # Generate LLM score
        evaluated_response = generate_llm_score(response)
        evaluated_responses.append(evaluated_response)
    
    # Calculate average LLM score
    valid_scores = [r['llm_score'] for r in evaluated_responses if 'llm_score' in r and 'llm_error' not in r]
    avg_llm_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    bt.logging.info(f"LLM evaluation completed. Average score: {avg_llm_score:.3f}")
    
    return evaluated_responses


def save_responses_for_judge(responses: List[Dict[str, Any]], output_file: str) -> str:
    """
    Save responses in JSONL format for external VoiceBench judge evaluation.
    
    Args:
        responses: List of response dictionaries
        output_file: Output file path
        
    Returns:
        Path to saved file
    """
    try:
        with open(output_file, 'w') as f:
            for response in responses:
                # Format for VoiceBench judge
                judge_item = {
                    'prompt': response.get('prompt', ''),
                    'response': response.get('response', ''),
                    'reference': response.get('reference', '')
                }
                f.write(json.dumps(judge_item) + '\n')
        
        bt.logging.info(f"Saved {len(responses)} responses to {output_file} for judge evaluation")
        return output_file
        
    except Exception as e:
        bt.logging.error(f"Error saving responses for judge: {e}")
        return ""


def calculate_llm_scores(dataset_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate LLM-based scores for dataset results.
    
    Args:
        dataset_results: Results from VoiceBench evaluation
        
    Returns:
        Dictionary of LLM scores per dataset
    """
    llm_scores = {}
    
    for dataset_key, dataset_result in dataset_results.items():
        if 'error' in dataset_result:
            llm_scores[dataset_key] = 0.0
            continue
            
        responses = dataset_result.get('responses', [])
        if not responses:
            llm_scores[dataset_key] = 0.0
            continue
        
        # Evaluate responses with LLM
        evaluated_responses = evaluate_responses_with_llm(responses)
        
        # Calculate average LLM score
        valid_scores = [r['llm_score'] for r in evaluated_responses if 'llm_score' in r and 'llm_error' not in r]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        llm_scores[dataset_key] = avg_score
        
        # Update dataset results with LLM scores
        dataset_result['responses'] = evaluated_responses
        dataset_result['llm_score'] = avg_score
        dataset_result['llm_valid_count'] = len(valid_scores)
    
    # Calculate overall LLM score
    if llm_scores:
        positive_scores = [s for s in llm_scores.values() if s > 0]
        if positive_scores:
            llm_scores['overall'] = sum(positive_scores) / len(positive_scores)
        else:
            llm_scores['overall'] = 0.0
    else:
        llm_scores['overall'] = 0.0
    
    return llm_scores