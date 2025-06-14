# Cross-Lingual Prompt Cycle Evaluator

A comprehensive tool for evaluating translation quality through round-trip translation analysis. This tool translates text from any source language to English and back, then measures how well the original meaning was preserved.

## Features

- **Multi-language Support**: Works with any language pair supported by Helsinki-NLP models
- **Comprehensive Metrics**: 
  - Character-level BLEU score
  - Token-level edit distance
  - Normalized explainability score (0-1)
- **Rich CLI Output**: Beautiful terminal interface with colored formatting
- **JSON Export**: Save results for further analysis
- **GPU Acceleration**: Automatic GPU detection and usage

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NLTK
- Rich
- python-Levenshtein

## 🛠Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cross-lingual-evaluator
```

2. Install required packages:
```bash
pip install torch transformers nltk python-Levenshtein rich
```

## Usage

### Basic Usage

Evaluate a Chinese prompt (default):
```bash
python prompt_cycle_evaluator.py "你好，世界"
```

### Specify Different Languages

Evaluate a French prompt:
```bash
python prompt_cycle_evaluator.py "Bonjour le monde" --lang fr
```

Evaluate a German prompt:
```bash
python prompt_cycle_evaluator.py "Hallo Welt" --lang de
```

Or for a more complex prompt:
```bash
ython "Cross-Lingual Prompt Cycle Evaluator.py" "Obwohl es den ganzen Tag geregnet hat, haben wir beschlossen, einen Spaziergang durch den Park zu machen, um die frische Luft zu genießen" --lang de
```

### Command Line Arguments

- `prompt`: The input text to evaluate (required)
- `--lang`: Source language code (default: "zh" for Chinese)

### Supported Language Codes

Common language codes include:
- `zh` - Chinese
- `fr` - French  
- `de` - German
- `es` - Spanish
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic

## Output Example

```
Cross-Lingual Prompt Cycle Evaluation
┌────────────────┬────────────────────┐
│ Step           │ Text               │
├────────────────┼────────────────────┤
│ Original       │ 你好，世界          │
│ English        │ Hello, world       │
│ Back Translated│ 你好，世界          │
└────────────────┴────────────────────┘

Evaluation Metrics
┌──────────────────────┬─────────┐
│ Metric               │ Score   │
├──────────────────────┼─────────┤
│ char_bleu           │ 0.8532  │
│ edit_distance       │ 0.0000  │
│ normalized_distance  │ 1.0000  │
│ explainability      │ 0.9266  │
└──────────────────────┴─────────┘

Translation Quality: Excellent (0.93)
```

## Metrics Explanation

### Character-level BLEU Score
Measures n-gram overlap between original and back-translated text at character level.

### Edit Distance
Levenshtein distance measuring character-level changes needed to transform one string into another.

### Normalized Edit Distance  
Edit distance normalized by maximum string length (higher is better).

### Explainability Score
Combined metric averaging BLEU and normalized edit distance, indicating overall meaning preservation:
- **0.8-1.0**: Excellent preservation
- **0.6-0.8**: Good preservation  
- **<0.6**: Poor preservation

## Output Files

Results are automatically saved to `evaluation_results.json`:

```json
{
  "original": "你好，世界",
  "english": "Hello, world",
  "back_translated": "你好，世界",
  "metrics": {
    "char_bleu": 0.8532,
    "edit_distance": 0.0,
    "normalized_edit_distance": 1.0,
    "explainability": 0.9266
  }
}
```

## Technical Details

- **Translation Models**: Uses Helsinki-NLP Opus-MT models
- **Tokenization**: NLTK word tokenization
- **GPU Support**: Automatically detects and uses GPU when available
- **Model Caching**: Downloaded models are cached for subsequent runs

## Notes

- First run will download required models and NLTK data
- Models are automatically cached after first download
- Supports any language pair available in Helsinki-NLP model collection
- GPU usage significantly speeds up translation for longer texts

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Helsinki-NLP for providing the translation models
- Hugging Face Transformers library
- NLTK for tokenization tools
