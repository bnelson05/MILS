# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Text, Dict, Tuple, List
import transformers
import torch
import numpy as np
from collections import OrderedDict
from random import sample
import re


def strip_line_counters(text):
    """
    Extract clean image descriptions from model output.
    
    Handles:
    - Old style: 1. Description
    - New style: **1\.** on one line, Description on next line
    - Inline style: **1\.** Description on same line
    """
    cleaned_lines = []
    lines = text.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if this line is just a number marker: **1\.** or 1. or **1.**
        if re.match(r'^\**\d+\\?\.?\**\s*$', line):
            # Next line should be the description
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 10:  # Minimum length check
                    # Clean up the description
                    desc = next_line.strip('"').strip("'").replace('**', '').strip()
                    # Skip if it contains meta-commentary
                    if not any(bad in desc.lower() for bad in ['here are', 'five more', 'summary', 'summaries', 'entry combines', 'these entries']):
                        cleaned_lines.append(desc)
                i += 2  # Skip both the marker and description line
                continue
        
        # Check if number and description are on same line: 1. Description or **1\.** Description
        match = re.match(r'^\**\d+\\?\.?\**[\s\-]*["\*\-]*\s*(.+)', line)
        if match:
            desc = match.group(1).strip()
            desc = desc.strip('"').strip("'").replace('**', '').strip()
            # Must be at least 10 chars and not meta-commentary
            if len(desc) > 10:
                if not any(bad in desc.lower() for bad in ['here are', 'five more', 'summary', 'summaries', 'entry combines']):
                    cleaned_lines.append(desc)
            i += 1
            continue
        
        i += 1
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for line in cleaned_lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    
    return result

def extract_descriptions_from_gpt2(text):
    """Enhanced extraction specifically for GPT-2 outputs that may not follow perfect format"""
    cleaned_lines = []
    lines = text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Clean up the line
        line = line.replace("<|endoftext|>", "").replace("<pad>", "").strip()
        
        # Try to extract numbered descriptions (1. 2. etc.)
        if ". " in line[:5]:
            period_index = line.find(".")
            description = line[period_index + 2:].strip()
        # Try to extract descriptions with other patterns
        elif line.startswith("- "):
            description = line[2:].strip()
        elif line.startswith("* "):
            description = line[2:].strip()
        # Try to extract from colon patterns (common in GPT-2 outputs)
        elif ": " in line:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                description = parts[1].strip()
            else:
                description = line
        else:
            # If it looks like a reasonable description, use it
            if (len(line.split()) > 2 and len(line) < 100 and 
                any(word in line.lower() for word in ['image', 'picture', 'photo', 'pattern', 'texture', 'showing', 'with', 'of'])):
                description = line
            else:
                continue
        
        # Clean up the description
        description = description.split("(")[0].strip()  # Remove parenthetical content
        description = description.rstrip(".,!?;:")  # Remove trailing punctuation
        
        # Filter out obviously bad descriptions
        if (description and 
            len(description.split()) >= 3 and  # At least 3 words
            len(description) < 100 and  # Not too long
            description not in cleaned_lines and  # Not duplicate
            not description.lower().startswith(('example', 'note', 'remember', 'if you', 'you can', 'this is'))):  # Filter out instruction text
            cleaned_lines.append(description)
    
    # If we didn't find any good descriptions, try a more aggressive approach
    if not cleaned_lines:
        # Look for any line that mentions visual concepts
        for line in lines:
            line = line.strip()
            if (line and len(line.split()) >= 3 and 
                any(word in line.lower() for word in ['image', 'picture', 'photo', 'visual', 'pattern', 'texture', 'color', 'shape'])):
                # Take up to the first sentence
                description = line.split('.')[0].strip()
                if description and len(description.split()) >= 3:
                    cleaned_lines.append(description)
    
    return set(cleaned_lines)


class Generator(object):
    def __init__(
        self,
        text_pipeline,
        text_model_name,
        requested_number: int = 100,
        keep_previous: int = 100,
        prompt: Text = "",
        key=lambda x: x[0],
        batch_size: int = 1,
        max_new_tokens: int = 3000,
        device: Text = "cuda:0",
        post_text_function=None,
        verbose: int = 1,
        exploration: float = 0.0,
    ):
        self.key = key
        self.text_model_name = text_model_name
        self.exploration = exploration
        self.batch_size = batch_size
        self.keep_previous = keep_previous
        self.max_new_tokens = max_new_tokens
        self.requested_number = requested_number
        self.prompt = prompt
        self.text_pipeline = text_pipeline
        self.device = device
        self.terminators = [
            self.text_pipeline.tokenizer.eos_token_id,
            self.text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        if "llama" in text_model_name:
            self.text_pipeline.tokenizer.pad_token_id = (
                self.text_pipeline.model.config.eos_token_id[0]
            )
        self.post_text_function = post_text_function
        self.verbose = verbose
        self._post_text_cached = {}

    def _cached_post_text(self, text_set):
        results = [None for i in text_set]
        needed_indices = []
        to_run = []
        for i, text in enumerate(text_set):
            if text in self._post_text_cached:
                results[i] = self._post_text_cached[text]
            else:
                needed_indices.append(i)
                to_run.append(text)
        if to_run:
            new_results = self.post_text_function(to_run)
            assert len(new_results) == len(needed_indices)
            for i, nr in zip(needed_indices, new_results):
                self._post_text_cached[text_set[i]] = nr
                results[i] = nr
        return results

    def _get_descriptions(self, task, list_of_pairs, **kwargs):
        """Converts the list of pairs into a text description that can be passed to the model"""
        descriptions = ""
        lines = set()
        for key, v in list_of_pairs:
            # import pdb; pdb.set_trace()
            if isinstance(key, float) or isinstance(key, np.float32):
                descriptions += f"{float(key):.3f}"
            else:
                # A list of keys
                print(key)
                for k in key:
                    descriptions += f"{float(k):.3f}, "
                descriptions = descriptions[:-2]
            if isinstance(v, tuple):
                v = v[0]  # This is in case we have a (text, image), comping from scorer
            lines.add(v)
            descriptions += f": {v}\n"
        new_prompt = self.prompt.replace("{descriptions}", descriptions).replace(
            "{requested_number}", str(self.requested_number)
        )
        for k, v in kwargs.items():
            new_prompt = new_prompt.replace("{" + k + "}", v[task])
        if self.verbose > 0:
            print(new_prompt)
        return new_prompt, lines

    def _get_requests_and_messages(self, task_dict, **kwargs):
        messages = []
        requests = []
        lines_set = []
        for task, value in task_dict.items():
            requests.append(task)
            assert len(value)
            assert len(value[0]) == 2, value[0]
            list_of_pairs = self.sort_with_exploration(value)
            new_prompt, lines = self._get_descriptions(task, list_of_pairs, **kwargs)
            lines_set.append(lines)
            if "llama" in self.text_model_name:
                messages += [
                    self.text_pipeline.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": new_prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                ]
            else:
                # For non-llama models like GPT-2, use raw text instead of chat format
                messages.append(new_prompt)
        return requests, messages, lines_set

    def _run_post_text_function(self, descriptions_per_caption):
        data = []
        for j in range(0, len(descriptions_per_caption), self.batch_size):
            # data can be a list of images
            data += self._cached_post_text(
                descriptions_per_caption[j : j + self.batch_size]
            )
        assert len(data) == len(descriptions_per_caption)
        return data

    def sort_with_exploration(self, values):
        if not self.exploration or self.keep_previous >= len(values):
            return sorted(values, key=self.key)[: self.keep_previous]
        s = sorted(values, key=self.key)
        data = s[: int(self.keep_previous * (1 - self.exploration))]
        return data + sample(
            s[int(self.keep_previous * (1 - self.exploration)) :],
            int(self.keep_previous * self.exploration),
        )

    def run_on_message_batch(self, current_messages_batch):
        """Generate text with model-specific optimizations but simple extraction"""
        
        # Fix tokenizer padding
        if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
            original_padding_side = self.text_pipeline.tokenizer.padding_side
            self.text_pipeline.tokenizer.padding_side = 'left'
        
        model_type = self.text_model_name.lower()
        
        # Model-specific generation parameters (KEEP FROM FORK)
        if "qwen" in model_type:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=180,
                do_sample=True,
                temperature=0.5,
                top_p=0.8,
                top_k=30,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )
        elif "gpt" in model_type:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        elif "llama" in model_type or "mistral" in model_type:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=min(self.max_new_tokens, 300),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        else:
            # Default parameters
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=min(self.max_new_tokens, 250),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False,
                truncation=True,
                pad_token_id=getattr(self.text_pipeline.tokenizer, 'pad_token_id', 0),
                eos_token_id=getattr(self.text_pipeline.tokenizer, 'eos_token_id', 0),
                repetition_penalty=1.1,
            )

        # ✅ DEBUG: print raw outputs immediately after generation
        print("\n=== RAW MODEL OUTPUTS ===")
        print(f"Model: {self.text_model_name}")
        for i, out in enumerate(outputs):
            print(f"Output {i}: {out}")
        print("=== END RAW ===\n")

        print("=== DEBUG PIPELINE OUTPUT ===")
        print(outputs)
        print("=== END DEBUG ===")
        
        # SIMPLE EXTRACTION (FROM ORIGINAL REPO)
        results = []
        for text in outputs:
            # Extract generated text
            generated_text = ""
            if isinstance(text, list) and len(text) > 0:
                if isinstance(text[0], dict) and "generated_text" in text[0]:
                    generated_text = text[0]["generated_text"]
                else:
                    generated_text = str(text[0])
            elif isinstance(text, dict) and "generated_text" in text:
                generated_text = text["generated_text"]
            else:
                generated_text = str(text)
            
            # Use simple strip_line_counters from original repo
            extracted_descriptions = strip_line_counters(generated_text)
            
            # Minimal fallback
            if len(extracted_descriptions) == 0:
                extracted_descriptions = {"Generated image description"}
            
            results.append(extracted_descriptions)
            
            if self.verbose > 0:
                print(f"Model: {self.text_model_name}")
                print(f"Generated text: {generated_text[:150]}...")
                print(f"Extracted {len(extracted_descriptions)} descriptions")
        
        # Restore padding
        if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
            self.text_pipeline.tokenizer.padding_side = original_padding_side
        
        return results

    def __call__(self, task_dict: Dict[Text, List[Tuple[float, Text]]], **kwargs):
        """Task dict is a dictionary from filename to a list of tuples (float, txt)"""
        requests, messages, lines_set = self._get_requests_and_messages(
            task_dict, **kwargs
        )
        assert len(requests) == len(messages)
        # messages is ordered per task
        all_responses = []
        # Do the generation, in batches:
        for i in range(0, len(messages), self.batch_size):
            current_messages_batch = messages[i : i + self.batch_size]
            outputs = self.run_on_message_batch(current_messages_batch)
            # CHANGED
            texts = [list(set(text) - lines_set[i]) for i, text in enumerate(outputs)]
            # texts = [list(text - lines_set[i]) for i, text in enumerate(outputs)]

            assert len(texts) == len(current_messages_batch), (
                len(texts),
                len(current_messages_batch),
            )
            # texts is list of lists
            if self.post_text_function is not None:
                if self.verbose > 0:
                    print(texts)
                for descriptions_per_caption in texts:
                    all_responses.append(
                        list(
                            zip(
                                descriptions_per_caption,
                                self._run_post_text_function(descriptions_per_caption),
                            )
                        )
                    )
            else:
                if self.verbose > 0:
                    print(texts)
                all_responses += texts
            if self.verbose > 0:
                print(all_responses)
        assert len(all_responses) == len(requests), (len(all_responses), len(requests))
        new_data = {r: m for (r, m) in zip(requests, all_responses)}
        # Add the old data:
        for r, value in task_dict.items():
            list_of_pairs = sorted(value, key=self.key)[: self.keep_previous]
            for i in list_of_pairs:
                if i[1] not in new_data[r]:
                    new_data[r].append(i[1])
        return new_data
