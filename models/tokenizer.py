from typing import List

from torchtune.models.llama3._tokenizer import Llama3Tokenizer, LLAMA3_SPECIAL_TOKENS
from torchtune.modules.tokenizers._tiktoken import MAX_ENCODE_CHARS, MAX_NO_WHITESPACE_CHARS

# use special tokens from TikTokenTokenizer, add some for MM delimiters
START_IMAGE = "<|start_image|>"
END_IMAGE = "<|end_image|>"
START_VIDEO = "<|start_video|>"
END_VIDEO = "<|end_video|>"
START_AUDIO = "<|start_audio|>"
END_AUDIO = "<|end_audio|>"

A2A_SPECIAL_TOKENS = {
    **LLAMA3_SPECIAL_TOKENS,
    **{
        START_IMAGE: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_2|>"],
        END_IMAGE: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_3|>"],
        START_AUDIO: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_4|>"],
        END_AUDIO: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_5|>"],
        START_VIDEO: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_6|>"],
        END_VIDEO: LLAMA3_SPECIAL_TOKENS["<|reserved_special_token_7|>"],
    }
}
for i in [2, 3, 4, 5, 6, 7]:
    del A2A_SPECIAL_TOKENS[f"<|reserved_special_token_{i}|>"]


class A2ATokenizer(Llama3Tokenizer):
    def encode(
        self,
        text: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        """
        Encode a string into a list of token ids. Assumes that the string
        contains no special tokens.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the tokenizer's bos_id to the encoded string.
                Default True.
            add_eos (bool): Whether to add the tokenizer's eos_id to the encoded string.
                Default True.

        Returns:
            List[int]: The list of token ids.
        """
        substrs: List[str] = []
        tokens = []
        if not text:
            return []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = self.tt_model._split_long_repetitions(
                substr, MAX_NO_WHITESPACE_CHARS
            )
            substrs.extend(sliced_substr)
        for substr in substrs:
            # allowed_special and disallowed_special are used by tiktoken to define
            # how special tokens are encoded. Our setting here is to encode any
            # special token as regular text and prevent tiktoken from raising errors.
            # This means we should only call encode on strings not containing special tokens.
            tokens.extend(
                self.tt_model.tt_model.encode(
                    substr,
                    allowed_special=set(A2A_SPECIAL_TOKENS),
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens


def a2a_tokenizer(path: str):
    tiktoken = A2ATokenizer(path, special_tokens=A2A_SPECIAL_TOKENS)
    tiktoken.pad_id = 0
    return tiktoken
