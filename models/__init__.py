# from torchtune.models import convert_weights

# from models.tokenizer import a2a_tokenizer
# from models.mmllama3 import lora_mmllama3_8b, mmllama3_8b, imagebind_huge

# __all__ = [
#     "a2a_tokenizer",
#     "lora_mmllama3_8b",
#     "mmllama3_8b",
#     "imagebind_huge",
# ]

# _BASE_TRAINABLE = [
#     "tok_embeddings.proj_to_llama.0.weight",
#     "tok_embeddings.proj_to_llama.0.bias",
#     "tok_embeddings.proj_to_llama.2.weight",
#     "tok_embeddings.proj_to_llama.2.bias",
#     "tok_embeddings.proj_to_llama.3.weight",
#     "tok_embeddings.proj_to_llama.3.bias",
#     "output.proj_from_llama.0.weight",
#     "output.proj_from_llama.0.bias",
#     "output.proj_from_llama.2.weight",
#     "output.proj_from_llama.2.bias",
#     "output.proj_from_llama.3.weight",
#     "output.proj_from_llama.3.bias",
# ]

# def add_proj_convert_weights():
#     # extend _FROM_META torchtune -> meta mapping with new parameter names
#     # allow existing ckpt-save code to work without changes
#     convert_weights._FROM_META.update({a: a for a in _BASE_TRAINABLE})


