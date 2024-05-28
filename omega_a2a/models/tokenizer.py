from typing import List

from torchtune.modules.tokenizers import TikTokenTokenizer
from torchtune.modules.tokenizers._utils import _split_long_repetitions
from torchtune.modules.tokenizers._tiktoken import (
    MAX_ENCODE_CHARS,
    MAX_NO_WHITESPACE_CHARS,
    ALL_SPECIAL_TOKENS,
)
  

# use special tokens from TikTokenTokenizer, add some for MM delimiters
START_IMAGE = "<|start_image|>"
END_IMAGE = "<|end_image|>"
START_VIDEO = "<|start_video|>"
END_VIDEO = "<|end_video|>"
START_AUDIO = "<|start_audio|>"
END_AUDIO = "<|end_audio|>"

A2A_SPECIAL_TOKENS = ALL_SPECIAL_TOKENS[:-2] + [
    START_IMAGE,
    END_IMAGE,
    START_VIDEO,
    END_VIDEO,
    START_AUDIO,
    END_AUDIO,
] + ALL_SPECIAL_TOKENS[-2:]

# override to allow START_IMAGE, END_IMAGE to be encoded
class A2ATokenizer(TikTokenTokenizer):
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
            add_bos (bool): Whether to add the beginning of sequence token.
            add_eos (bool): Whether to add the end of sequence token.

        Returns:
            List[int]: The list of token ids.
        """
        substrs: List[str] = []
        tokens = []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = _split_long_repetitions(substr, MAX_NO_WHITESPACE_CHARS)
            substrs.extend(sliced_substr)
        for substr in substrs:
            # allowed_special and disallowed_special are used by tiktoken to define
            # how special tokens are encoded. Our setting here is to encode any
            # special token as regular text and prevent tiktoken from raising errors.
            # This means we should only call encode on strings not containing special tokens.
            tokens.extend(
                self.tt_model.encode(
                    substr,
                    allowed_special=set([
                        START_IMAGE,
                        END_IMAGE,
                        START_VIDEO,
                        END_VIDEO,
                        START_AUDIO,
                        END_AUDIO,
                    ]),
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens


def a2a_tokenizer(path: str) -> TikTokenTokenizer:
    tiktoken = A2ATokenizer(path, all_special_tokens=A2A_SPECIAL_TOKENS)
    tiktoken.pad_id = 0
    return tiktoken
