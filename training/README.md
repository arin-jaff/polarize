# AI Coach Training

This directory contains everything needed to fine-tune and deploy the AI fitness coach model for the Polarize training application.

## Directory Structure

```
training/
├── README.md                    # This file
├── ARCHITECTURE.md              # System architecture and data flow
├── TRAINING_GUIDE.md            # Complete training process documentation
├── DATA_FORMAT.md               # Training data format specification
├── JSON_OUTPUT_SPEC.md          # LLM output JSON schema specification
├── train_model.ipynb            # Colab notebook for fine-tuning
├── generate_training_data.py    # Script to generate synthetic training data
├── validate_data.py             # Script to validate training data format
└── example_data/                # Example training data files
    ├── coaching_conversations.jsonl
    ├── plan_modifications.jsonl
    └── weekly_plans.jsonl
```

## Quick Start

### 1. Generate Training Data

```bash
python generate_training_data.py --output ./training_data.jsonl --count 1000
```

### 2. Fine-tune on Google Colab

1. Open `train_model.ipynb` in Google Colab
2. Upload your `training_data.jsonl` file
3. Run all cells to fine-tune the model
4. Download the LoRA adapters

### 3. Deploy with Ollama

```bash
# Create the Modelfile
ollama create fitness-coach -f Modelfile

# Test the model
ollama run fitness-coach "My CTL is 45 and I'm feeling tired. What should I do?"
```

## Training Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generate Data  │────▶│   Fine-tune     │────▶│ Deploy Model    │
│  (synthetic +   │     │  (QLoRA on      │     │  (Ollama +      │
│   real logs)    │     │   Colab Pro)    │     │   FastAPI)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Model Specifications

- **Base Model**: Mistral 7B Instruct v0.3 (or Gemma 2B for faster inference)
- **Fine-tuning Method**: QLoRA (4-bit quantization + Low-Rank Adaptation)
- **Training Hardware**: A100 GPU (Colab Pro) or RTX 3090/4090
- **Training Time**: ~2-4 hours for 1000 examples

## Key Concepts

### Supervised Fine-Tuning (SFT)

We use SFT rather than RLHF because:
1. We have clear, structured outputs (JSON)
2. Reward modeling for training advice is complex
3. SFT is simpler and more data-efficient

### Hybrid Architecture

The model doesn't work alone - it's part of a hybrid system:

1. **Context Builder** (Python) - Fetches user data, builds structured prompts
2. **LLM** (Fine-tuned) - Generates structured JSON modifications
3. **Workout Modifier** (Python) - Validates and applies changes to database

This approach lets us:
- Keep the model focused on coaching decisions
- Validate outputs before applying
- Use deterministic code for data manipulation
- Iterate on prompts without retraining

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System design and component interactions |
| [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) | Step-by-step training instructions |
| [DATA_FORMAT.md](./DATA_FORMAT.md) | Training data schema and examples |
| [JSON_OUTPUT_SPEC.md](./JSON_OUTPUT_SPEC.md) | Expected model output format |

## Requirements

### For Training
- Google Colab Pro ($10/mo) or Pro+ ($50/mo)
- Or local GPU with 24GB+ VRAM

### For Deployment
- Ollama installed
- 8GB+ RAM for 7B model (4-bit quantized)
- 4GB+ RAM for 2B model

## Links

- [Ollama Documentation](https://ollama.com/docs)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
