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
from datetime import datetime
import os

class PromptCycleEvaluator:
    def __init__(self, save_to_file: bool = True, output_dir: str = "outputs"):
        """Initialize translation models and tokenizers"""
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('punkt_tab')  # Added this for newer NLTK versions
        
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if self.save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize console for terminal output
        self.console = Console()
        
        # Initialize file console if saving to file
        if self.save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(self.output_dir, f"evaluation_output_{timestamp}.txt")
            self.file_console = Console(file=open(self.output_file, "w", encoding="utf-8"), width=100)
        
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
    
    def display_report(self, prompt: str, metrics: Dict, english: str, back_translated: str, source_lang: str):
        """Display evaluation results in CLI and save to file"""
        # Create header with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"Cross-Lingual Prompt Cycle Evaluation - {timestamp}"
        
        # Print header to both console and file
        self.console.print(f"\n[bold blue]{header}[/bold blue]")
        if self.save_to_file:
            self.file_console.print(f"{header}")
            self.file_console.print("=" * len(header))
        
        # Create main results table
        table = Table(title="Translation Results")
        table.add_column("Step", style="cyan", width=15)
        table.add_column("Text", style="green", width=60)
        
        table.add_row("Original", prompt)
        table.add_row("English", english)
        table.add_row("Back Translated", back_translated)
        
        # Display to console
        self.console.print(table)
        
        # Save to file in a readable format
        if self.save_to_file:
            self.file_console.print(f"\nSource Language: {source_lang}")
            self.file_console.print(f"Original:        {prompt}")
            self.file_console.print(f"English:         {english}")
            self.file_console.print(f"Back Translated: {back_translated}")
        
        # Create metrics table
        metrics_table = Table(title="Evaluation Metrics")
        metrics_table.add_column("Metric", style="cyan", width=25)
        metrics_table.add_column("Score", style="green", width=15)
        
        for metric, value in metrics.items():
            metrics_table.add_row(metric, f"{value:.4f}")
            
        # Display metrics to console
        self.console.print(metrics_table)
        
        # Save metrics to file
        if self.save_to_file:
            self.file_console.print(f"\nEvaluation Metrics:")
            self.file_console.print("-" * 30)
            for metric, value in metrics.items():
                self.file_console.print(f"{metric:<25}: {value:.4f}")
        
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
            
        quality_msg = f"Translation Quality: {quality} ({score:.2f})"
        rprint(f"\n[{color}]{quality_msg}[/{color}]")
        
        if self.save_to_file:
            self.file_console.print(f"\n{quality_msg}")
            self.file_console.print("\n" + "="*50 + "\n")
    
    def close_file(self):
        """Close the file console"""
        if self.save_to_file and hasattr(self, 'file_console'):
            self.file_console.file.close()
            print(f"\nOutput saved to: {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual Prompt Cycle Evaluator")
    parser.add_argument("prompt", help="Input prompt to evaluate")
    parser.add_argument("--lang", default="zh", help="Source language code (default: zh)")
    parser.add_argument("--no-file", action="store_true", help="Don't save output to file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for files (default: outputs)")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PromptCycleEvaluator(
        save_to_file=not args.no_file,
        output_dir=args.output_dir
    )
    
    try:
        # Run evaluation
        metrics, english, back_translated = evaluator.evaluate_prompt(args.prompt, args.lang)
        evaluator.display_report(args.prompt, metrics, english, back_translated, args.lang)
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "source_language": args.lang,
            "original": args.prompt,
            "english": english,
            "back_translated": back_translated,
            "metrics": metrics
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"JSON results saved to: {json_filename}")
        
    finally:
        # Always close the file
        evaluator.close_file()

if __name__ == "__main__":
    main()
