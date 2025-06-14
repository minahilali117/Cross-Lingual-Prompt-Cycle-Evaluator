import argparse
from typing import Tuple, Dict
from transformers import pipeline, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import Levenshtein
import nltk
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import torch

class PromptCycleEvaluator:
    def __init__(self):
        """Initialize translation models and tokenizers"""
        # Download required NLTK data
        nltk.download('punkt')
        
        self.console = Console()
        
        # Initialize translation models
        self.models = {}
        self.tokenizers = {}
        
    def load_language_pair(self, source_lang: str, target_lang: str = "en"):
        """Load translation models for a language pair"""
        # Load forward translation model (source -> English)
        model_name_forward = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.models[f"{source_lang}-{target_lang}"] = pipeline(
            "translation", 
            model=model_name_forward,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load backward translation model (English -> source)
        model_name_backward = f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}"
        self.models[f"{target_lang}-{source_lang}"] = pipeline(
            "translation",
            model=model_name_backward,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load tokenizers
        self.tokenizers[source_lang] = AutoTokenizer.from_pretrained(model_name_forward)
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages"""
        model_key = f"{source_lang}-{target_lang}"
        if model_key not in self.models:
            self.load_language_pair(source_lang, target_lang)
            
        result = self.models[model_key](text)[0]['translation_text']
        return result
    
    def compute_metrics(self, original: str, back_translated: str, source_lang: str) -> Dict:
        """Compute comparison metrics between original and back-translated text"""
        # Tokenize texts
        original_tokens = word_tokenize(original)
        back_tokens = word_tokenize(back_translated)
        
        # Calculate character-level BLEU
        char_bleu = sentence_bleu(
            [list(original)],
            list(back_translated),
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        
        # Calculate token-level edit distance
        edit_distance = Levenshtein.distance(original, back_translated)
        max_length = max(len(original), len(back_translated))
        normalized_edit_distance = 1 - (edit_distance / max_length)
        
        # Calculate explainability score (average of BLEU and normalized edit distance)
        explainability = (char_bleu + normalized_edit_distance) / 2
        
        return {
            'char_bleu': char_bleu,
            'edit_distance': edit_distance,
            'normalized_edit_distance': normalized_edit_distance,
            'explainability': explainability
        }
    
    def evaluate_prompt(self, prompt: str, source_lang: str) -> Tuple[Dict, str, str]:
        """Perform round-trip translation and evaluation"""
        # Forward translation (source -> English)
        english = self.translate(prompt, source_lang, "en")
        
        # Back translation (English -> source)
        back_translated = self.translate(english, "en", source_lang)
        
        # Compute metrics
        metrics = self.compute_metrics(prompt, back_translated, source_lang)
        
        return metrics, english, back_translated
    
    def display_report(self, prompt: str, metrics: Dict, english: str, back_translated: str):
        """Display evaluation results in CLI"""
        table = Table(title="Cross-Lingual Prompt Cycle Evaluation")
        
        table.add_column("Step", style="cyan")
        table.add_column("Text", style="green")
        
        table.add_row("Original", prompt)
        table.add_row("English", english)
        table.add_row("Back Translated", back_translated)
        
        self.console.print(table)
        
        # Print metrics
        metrics_table = Table(title="Evaluation Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="green")
        
        for metric, value in metrics.items():
            metrics_table.add_row(metric, f"{value:.4f}")
            
        self.console.print(metrics_table)
        
        # Print explainability interpretation
        score = metrics['explainability']
        if score >= 0.8:
            color = "green"
            quality = "Excellent"
        elif score >= 0.6:
            color = "yellow"
            quality = "Good"
        else:
            color = "red"
            quality = "Poor"
            
        rprint(f"\n[{color}]Translation Quality: {quality} ({score:.2f})[/{color}]")

def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual Prompt Cycle Evaluator")
    parser.add_argument("prompt", help="Input prompt to evaluate")
    parser.add_argument("--lang", default="zh", help="Source language code (default: zh)")
    args = parser.parse_args()
    
    evaluator = PromptCycleEvaluator()
    metrics, english, back_translated = evaluator.evaluate_prompt(args.prompt, args.lang)
    evaluator.display_report(args.prompt, metrics, english, back_translated)
    
    # Save results to JSON
    results = {
        "original": args.prompt,
        "english": english,
        "back_translated": back_translated,
        "metrics": metrics
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()