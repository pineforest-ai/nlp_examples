from typing import Tuple, List, Optional, Union, Dict, Set, Any
import torch
from transformers import AutoTokenizer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('context_preprocessor')
class ContextPreprocessor(Component):
    def __init__(self, vocab_file: str, do_lower_case: bool = True, max_seq_length: int = 512, **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case, **kwargs)
        self.tokenizer.pad_token_id = 0

    def __call__(self, context_batch: List[List[str]], response_batch: List[str] = None) -> Dict[str, torch.Tensor]:
        lengths = []
        pr_context_batch = []
        for context in context_batch:
            if isinstance(context, list):
                context = " ".join(context)
            pr_context_batch.append(context)
            lengths.append(len(self.tokenizer.encode(context)))
        max_context_length = max(lengths)
        context_enc = self.tokenizer.batch_encode_plus(pr_context_batch, padding="max_length",
                                                       max_length=max_context_length, return_tensors="pt")
        if response_batch is None:
            return context_enc["input_ids"], context_enc["attention_mask"]
        else:
            lengths = []
            pr_response_batch = []
            for response in response_batch:
                if isinstance(response, list):
                    response = " ".join(response)
                pr_response_batch.append(response)
                lengths.append(len(self.tokenizer.encode(response)))
            max_response_length = max(lengths)
            response_enc = self.tokenizer.batch_encode_plus(pr_response_batch, padding="max_length",
                                                            max_length=max_response_length, return_tensors="pt")
            return context_enc["input_ids"], context_enc["attention_mask"], response_enc["input_ids"]
