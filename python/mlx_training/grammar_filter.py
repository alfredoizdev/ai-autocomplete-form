"""
Grammar quality filtering for training data preparation.
Uses language_tool_python for grammar checking and perplexity scoring.
"""

import language_tool_python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from typing import Tuple, List, Optional
import re

class GrammarFilter:
    def __init__(self):
        # Initialize grammar checker
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        
        # Initialize perplexity scorer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        # Grammar error thresholds
        self.max_grammar_errors = 1  # Allow max 1 minor error
        self.max_perplexity = 50.0  # Reject high perplexity text
        
    def check_grammar_quality(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Check grammar quality of text.
        Returns: (is_acceptable, quality_score, errors)
        """
        # Check for grammar errors
        matches = self.grammar_tool.check(text)
        
        # Filter out minor issues like missing commas
        serious_errors = []
        for match in matches:
            if match.ruleId not in ['COMMA_PARENTHESIS_WHITESPACE', 'WHITESPACE_RULE']:
                serious_errors.append(f"{match.ruleId}: {match.message}")
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(text)
        
        # Quality score (0-1)
        grammar_score = max(0, 1 - (len(serious_errors) / 3))
        perplexity_score = max(0, 1 - (perplexity / 100))
        quality_score = (grammar_score * 0.7 + perplexity_score * 0.3)
        
        is_acceptable = (
            len(serious_errors) <= self.max_grammar_errors and
            perplexity <= self.max_perplexity
        )
        
        return is_acceptable, quality_score, serious_errors
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text using GPT-2."""
        encodings = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
        return min(perplexity, 1000.0)  # Cap at 1000
    
    def validate_prompt_completion(self, prompt: str, completion: str) -> Tuple[bool, float]:
        """
        Validate that prompt + completion forms grammatically correct text.
        """
        # Check individual parts
        prompt_ok, prompt_score, _ = self.check_grammar_quality(prompt)
        
        # Check combined sentence
        combined = prompt + ' ' + completion
        combined_ok, combined_score, errors = self.check_grammar_quality(combined)
        
        # Ensure completion can stand alone grammatically
        # (starts with appropriate word)
        completion_start_ok = self._check_completion_start(prompt, completion)
        
        is_valid = prompt_ok and combined_ok and completion_start_ok
        quality = (prompt_score * 0.3 + combined_score * 0.7)
        
        return is_valid, quality
    
    def _check_completion_start(self, prompt: str, completion: str) -> bool:
        """Check if completion starts appropriately given the prompt."""
        # Get last word of prompt
        prompt_words = prompt.strip().split()
        if not prompt_words:
            return False
            
        last_word = prompt_words[-1].lower().rstrip(',')
        first_word = completion.strip().split()[0].lower() if completion.strip() else ""
        
        # Rules for appropriate completion starts
        if last_word in ['and', 'but', 'or', 'so', 'yet']:
            # After conjunctions, should continue with subject/verb
            return True
            
        if last_word in ['who', 'which', 'that', 'where', 'when', 'whose']:
            # After relative pronouns, should have verb/clause
            return True
            
        if last_word.endswith(','):
            # After comma, various continuations are OK
            return True
            
        if last_word in ['for', 'with', 'to', 'in', 'about', 'from']:
            # After prepositions, need noun/gerund
            return not first_word in ['is', 'are', 'was', 'were']
            
        # Default: completion should not start with lowercase unless continuation
        if completion[0].islower() and first_word not in ['and', 'but', 'or', 'who', 'which', 'that']:
            return False
            
        return True

def filter_training_data(data_pairs: List[Tuple[str, str]], 
                        min_quality: float = 0.7) -> List[Tuple[str, str]]:
    """
    Filter training data pairs for grammar quality.
    """
    filter = GrammarFilter()
    filtered_pairs = []
    
    for prompt, completion in data_pairs:
        is_valid, quality = filter.validate_prompt_completion(prompt, completion)
        
        if is_valid and quality >= min_quality:
            filtered_pairs.append((prompt, completion))
            
    return filtered_pairs