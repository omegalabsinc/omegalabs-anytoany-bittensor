# Testing your Sandboxed Model Container

Before getting started, make sure you have the following:
- An instance with a GPU
- The model that you want to test uploaded to HuggingFace
- A Bittensor wallet

## Run the scoring script

```
python neurons/scoring_manager.py --hotkey "{hotkey of your wallet}" --hf_repo_id {your huggingface repo id} --competition_id {your competition id} --block {your block number} --wandb.off --offline
```

Now, just wait for the test_scoring_api.py script to finish, and at the end, you will see your model score in the terminal and score saved to `scoring_task_result.json`.
