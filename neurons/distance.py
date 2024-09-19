import torch

def levenshtein_distance(generated_tokens, gt_tokens, device, dtype):
    if len(generated_tokens) < len(gt_tokens):
        return levenshtein_distance(gt_tokens, generated_tokens, device, dtype)

    if len(gt_tokens) == 0:
        return len(generated_tokens)

    previous_row = torch.arange(len(gt_tokens) + 1, dtype=dtype, device=device)
    for char_generated in generated_tokens:
        current_row = torch.zeros(len(gt_tokens) + 1, dtype=dtype, device=device)
        current_row[0] = previous_row[0] + 1
        current_row[1:] = torch.min(
            torch.stack([
                previous_row[1:] + 1,
                current_row[:-1] + 1,
                previous_row[:-1] + (char_generated != torch.tensor(gt_tokens, dtype=dtype, device=device))
            ]),
            dim=0
        ).values
        previous_row = current_row

    return previous_row[-1].item()


def calculate_length_difference_score(generated_tokens, gt_tokens, steepness=2, device=None, dtype=None):
    generated_len = len(generated_tokens)
    gt_len = len(gt_tokens)
    length_difference = abs(generated_len - gt_len)
    
    penalty = 1 - torch.exp(
                -torch.tensor(length_difference, dtype=dtype, device=device) * steepness / gt_len
            ).item()
    
    return penalty

def levenshtein_distance_with_length_penalty(generated_tokens, gt_tokens, length_diff_penalty_steepness, device, dtype):
    base_distance = levenshtein_distance(generated_tokens, gt_tokens, device, dtype)
    length_score = calculate_length_difference_score(generated_tokens, gt_tokens, length_diff_penalty_steepness, device, dtype)
    return base_distance * (1 - length_score)
