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
    # Split the text into lines
    lines = text.split("\n")

    # Strip the line counters and collect the cleaned lines
    cleaned_lines = []
    for line in lines:
        # Check if the line contains a period followed by a space, which is expected after the counter
        if ". " in line[:5]:  # A huristic to only count
            # Find the first period which is used to separate the counter from the text
            period_index = line.find(".")
            # Extract the text after the period and the space
            cleaned_line = line[period_index + 2 :]
            text = (
                cleaned_line.split("(")[0]
                .strip()
                .replace("<|endoftext|>", "")
                .replace("<pad>", "")
            )
            text = text.replace("!", "")
            if text and text not in cleaned_lines:
                cleaned_lines.append(text)

    return set(cleaned_lines)


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
        """Generate text with model-specific optimizations"""
        
        # Fix tokenizer padding for decoder-only models
        if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
            original_padding_side = self.text_pipeline.tokenizer.padding_side
            self.text_pipeline.tokenizer.padding_side = 'left'
        
        # Detect model type for optimization
        model_type = self.text_model_name.lower()
        
        # DialoGPT-specific optimization (conversational model)
        if "dialogo" in model_type or "dialogpt" in model_type:
            # DialoGPT needs special handling for list generation
            modified_prompts = []
            for prompt in current_messages_batch:
                # Add explicit instruction for list format
                modified_prompt = f"{prompt}\n\nPlease generate exactly 12 numbered descriptions (1. 2. 3. etc.):\n1."
                modified_prompts.append(modified_prompt)
            
            outputs = self.text_pipeline(
                modified_prompts,
                batch_size=len(modified_prompts),
                max_new_tokens=250,  # Increased for DialoGPT
                do_sample=True,
                temperature=0.6,  # Lower for more focused output
                top_p=0.8,
                top_k=30,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Higher to avoid repetition
                no_repeat_ngram_size=2,  # Prevent immediate repetition
            )
            
            # Enhanced extraction for DialoGPT conversational style
            results = []
            for text in outputs:
                generated_text = ""
                if isinstance(text, list) and len(text) > 0:
                    generated_text = text[0].get("generated_text", "")
                elif isinstance(text, dict):
                    generated_text = text.get("generated_text", "")
                else:
                    generated_text = str(text)
                
                # DialoGPT-specific parsing
                extracted_descriptions = self.extract_dialogo_descriptions(generated_text)
                
                # If still low, try fallback
                if len(extracted_descriptions) < 3:
                    fallback_descriptions = self.extract_descriptions_fallback(generated_text)
                    extracted_descriptions.update(fallback_descriptions)
                
                # Ensure minimum descriptions
                if len(extracted_descriptions) == 0:
                    extracted_descriptions = {"Image with visual elements", "Picture showing objects", "Photo displaying content"}
                
                results.append(extracted_descriptions)
            
            # Restore original padding
            if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
                self.text_pipeline.tokenizer.padding_side = original_padding_side
                
            return results
        
        # Qwen-specific optimization (instruction-tuned model)
        elif "qwen" in model_type:
            # Qwen models work best with specific parameters
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=180,  # Optimized for numbered lists
                do_sample=True,
                temperature=0.5,  # Lower temperature for more focused output
                top_p=0.8,  # More focused sampling
                top_k=30,  # Limit vocabulary for cleaner output
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Prevent repetition
                no_repeat_ngram_size=2,
            )
            
            # Qwen-specific text extraction
            results = []
            for text in outputs:
                generated_text = ""
                if isinstance(text, list) and len(text) > 0:
                    generated_text = text[0].get("generated_text", "")
                elif isinstance(text, dict):
                    generated_text = text.get("generated_text", "")
                else:
                    generated_text = str(text)
                
                # Handle chat template responses
                if "assistant" in generated_text:
                    generated_text = generated_text.split("assistant")[-1]
                
                # Qwen-specific extraction
                extracted_descriptions = self.extract_qwen_descriptions(generated_text)
                
                # Fallback if needed
                if len(extracted_descriptions) < 3:
                    fallback_descriptions = self.extract_descriptions_robust(generated_text)
                    extracted_descriptions.update(fallback_descriptions)
                
                results.append(extracted_descriptions)
            
            # Restore padding
            if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
                self.text_pipeline.tokenizer.padding_side = original_padding_side
                
            return results
        
        # LLaMA-style models (with chat templates)
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
            
            # Extract text from chat-style models
            results = []
            for text in outputs:
                generated_text = ""
                if isinstance(text, list) and len(text) > 0:
                    generated_text = text[0].get("generated_text", "")
                elif isinstance(text, dict):
                    generated_text = text.get("generated_text", "")
                else:
                    generated_text = str(text)
                
                # Handle chat template responses
                if "assistant" in generated_text:
                    generated_text = generated_text.split("assistant")[-1]
                
                extracted_descriptions = self.extract_descriptions_robust(generated_text)
                results.append(extracted_descriptions)
            
            # Restore padding
            if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
                self.text_pipeline.tokenizer.padding_side = original_padding_side
                
            return results
        
        # GPT-2 style models  
        elif "gpt" in model_type:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=200,  # Reduced for GPT-2
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        
        # MobiLlama and other small models
        elif "mobillama" in model_type or "phi" in model_type:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size=len(current_messages_batch),
                max_new_tokens=150,  # Smaller for efficiency
                do_sample=True,
                temperature=0.8,  # Higher temperature for small models
                top_p=0.9,
                return_full_text=False,
                truncation=True,
                pad_token_id=self.text_pipeline.tokenizer.pad_token_id,
                eos_token_id=self.text_pipeline.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Default parameters for unknown models
        else:
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
        
        # Enhanced text extraction with better debugging for non-DialoGPT models
        results = []
        for i, text in enumerate(outputs):
            generated_text = ""
            
            if isinstance(text, list):
                if len(text) > 0 and isinstance(text[0], dict) and "generated_text" in text[0]:
                    generated_text = text[0]["generated_text"]
                elif len(text) > 0:
                    generated_text = str(text[0])
            elif isinstance(text, dict) and "generated_text" in text:
                generated_text = text["generated_text"]
            else:
                generated_text = str(text)
            
            # Enhanced extraction with better parsing
            extracted_descriptions = self.extract_descriptions_robust(generated_text)
            
            # If we got very few descriptions, try alternative parsing
            if len(extracted_descriptions) < 3:
                fallback_descriptions = self.extract_descriptions_fallback(generated_text)
                extracted_descriptions.update(fallback_descriptions)
            
            # Ensure we have at least some descriptions
            if len(extracted_descriptions) == 0:
                extracted_descriptions = {"Generated image description"}
            
            results.append(extracted_descriptions)
            
            if self.verbose > 0:
                print(f"Model: {self.text_model_name}")
                print(f"Generated text: {generated_text[:150]}...")
                print(f"Extracted {len(extracted_descriptions)} descriptions")
        
        # Restore original padding
        if hasattr(self.text_pipeline.tokenizer, 'padding_side'):
            self.text_pipeline.tokenizer.padding_side = original_padding_side
        
        return results
    
    def extract_qwen_descriptions(self, text):
        """Specialized extraction for Qwen instruction-tuned model outputs"""
        descriptions = set()
        
        # Clean the text first
        text = text.replace("```", "").replace("**", "").strip()
        
        # Split into lines for processing
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove Qwen's common prefixes and formatting
            line = re.sub(r'^(Here are|Here\'s|Based on|Generate|Additional|The following)', '', line, flags=re.IGNORECASE).strip()
            line = line.replace("description:", "").replace("Description:", "").strip()
            
            # Pattern 1: Clean numbered lists (primary target)
            if re.match(r'^\d+[\.\)]\s+', line):
                desc = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                # Clean up common Qwen artifacts
                desc = desc.replace('"', '').replace("'", "").strip()
                desc = re.sub(r'^\[.*?\]\s*', '', desc)  # Remove [description] format
                if self.is_valid_qwen_description(desc):
                    descriptions.add(desc)
            
            # Pattern 2: Look for descriptions in quotes or brackets
            elif '"' in line or "'" in line:
                # Extract quoted content
                matches = re.findall(r'["\']([^"\']*)["\']', line)
                for match in matches:
                    if self.is_valid_qwen_description(match):
                        descriptions.add(match)
            
            # Pattern 3: Lines that start with visual words (but aren't instructions)
            elif (any(word in line.lower()[:20] for word in ['image', 'picture', 'photo', 'scene']) and
                  not any(bad in line.lower() for bad in ['generate', 'write', 'create', 'you need', 'based on', 'following'])):
                if self.is_valid_qwen_description(line):
                    descriptions.add(line)
            
            # Pattern 4: Simple descriptive lines (fallback)
            elif (len(line.split()) >= 3 and len(line.split()) <= 12 and
                  any(word in line.lower() for word in ['with', 'showing', 'featuring', 'displaying']) and
                  not any(bad in line.lower() for bad in ['score', 'higher', 'better', 'maximize', 'instruction'])):
                if self.is_valid_qwen_description(line):
                    descriptions.add(line)
        
        # If we got very few descriptions, try extracting from structured lists
        if len(descriptions) < 3:
            # Look for any structured content that might be buried in formatting
            structured_content = re.findall(r'(\d+\.\s*[^.]+\.)', text)
            for content in structured_content:
                clean_content = re.sub(r'^\d+\.\s*', '', content).rstrip('.')
                if self.is_valid_qwen_description(clean_content):
                    descriptions.add(clean_content)
        
        return descriptions

    def is_valid_qwen_description(self, desc):
        """Validate descriptions specifically for Qwen output"""
        if not desc or len(desc.strip()) < 5:
            return False
        
        desc = desc.strip()
        words = desc.split()
        
        # Length check
        if len(words) < 3 or len(words) > 15:
            return False
        
        # Filter out instruction-like text (common in Qwen)
        bad_starts = [
            'generate', 'write', 'create', 'you need', 'i am', 'higher score', 
            'be creative', 'remember', 'try to', 'make sure', 'please', 'note that',
            'based on', 'according to', 'following', 'additional', 'here are',
            'description', 'short image', 'requirements', 'format'
        ]
        if any(desc.lower().startswith(bad) for bad in bad_starts):
            return False
        
        # Filter out meta-commentary
        bad_content = [
            'score', 'higher', 'better', 'maximize', 'criteria', 'guidelines',
            'instruction', 'template', 'example', 'format', 'output', 'generate'
        ]
        if any(bad in desc.lower() for bad in bad_content):
            return False
        
        # Must contain visual/descriptive words
        visual_words = [
            'image', 'picture', 'photo', 'showing', 'with', 'featuring', 
            'displaying', 'scene', 'view', 'pattern', 'texture', 'color'
        ]
        if not any(word in desc.lower() for word in visual_words):
            return False
        
        return True
    
    def extract_dialogo_descriptions(self, text):
        """Specialized extraction for DialoGPT conversational outputs"""
        descriptions = set()
        
        # DialoGPT tends to be more conversational, so look for different patterns
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove conversational markers
            line = line.replace("Sure!", "").replace("Here are", "").replace("I can", "").strip()
            
            # Pattern 1: Numbered list (most common)
            if re.match(r'^\d+[\.\)]\s*', line):
                desc = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if self.is_valid_description(desc):
                    descriptions.add(desc)
            
            # Pattern 2: Dash or bullet points
            elif line.startswith(('- ', '• ', '* ', '→ ')):
                desc = line[2:].strip()
                if self.is_valid_description(desc):
                    descriptions.add(desc)
            
            # Pattern 3: Conversational descriptions
            elif any(word in line.lower() for word in ['image', 'picture', 'photo', 'scene']):
                # Clean up conversational elements
                desc = line.replace("Here's", "").replace("This is", "").strip()
                if self.is_valid_description(desc):
                    descriptions.add(desc)
        
        return descriptions
    
    def extract_descriptions_robust(self, text):
        """Robust extraction for GPT-2 outputs"""
        descriptions = set()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Clean the line
            line = line.replace("<|endoftext|>", "").replace("<pad>", "").strip()
            
            # Pattern 1: Numbered list (1. 2. etc.)
            if re.match(r'^\d+\.', line):
                desc = re.sub(r'^\d+\.\s*', '', line).strip()
                if self.is_valid_description(desc):
                    descriptions.add(desc)
            
            # Pattern 2: Bullet points
            elif line.startswith(('- ', '• ', '* ')):
                desc = line[2:].strip()
                if self.is_valid_description(desc):
                    descriptions.add(desc)
            
            # Pattern 3: Lines that look like descriptions
            elif self.is_valid_description(line):
                descriptions.add(line)
        
        return descriptions
    
    def extract_descriptions_fallback(self, text):
        """Fallback extraction for difficult cases"""
        descriptions = set()
        
        # Try to extract any sentence that mentions visual elements
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and 
                len(sentence.split()) >= 3 and 
                len(sentence) < 80 and
                any(word in sentence.lower() for word in 
                    ['image', 'picture', 'photo', 'shows', 'with', 'pattern', 'texture', 'color', 'object'])):
                descriptions.add(sentence)
        
        return descriptions
    
    def is_valid_description(self, desc):
        """Check if a description is valid"""
        if not desc or len(desc) < 3:
            return False
        
        words = desc.split()
        if len(words) < 3 or len(words) > 15:
            return False
        
        # Filter out instruction text
        bad_starts = ['generate', 'write', 'create', 'you need', 'i am', 'higher score', 'be creative']
        if any(desc.lower().startswith(bad) for bad in bad_starts):
            return False
        
        return True

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
            texts = [list(text - lines_set[i]) for i, text in enumerate(outputs)]

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
