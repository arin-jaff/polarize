# Complete AI Coach Training Guide

This guide covers every step of training the fitness coach AI model, from data preparation to deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding the Training Process](#understanding-the-training-process)
3. [Preparing Training Data](#preparing-training-data)
4. [Training on Google Colab](#training-on-google-colab)
5. [Training Locally (Optional)](#training-locally-optional)
6. [Creating the Ollama Model](#creating-the-ollama-model)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Deployment](#deployment)
9. [Iterating and Improving](#iterating-and-improving)

---

## Prerequisites

### Required Accounts
- Google account with Colab access
- Hugging Face account (for model access)

### Recommended
- Colab Pro ($10/mo) or Pro+ ($50/mo) for faster GPUs
- Ollama installed locally

### Knowledge
- Basic Python
- Understanding of training metrics (TSS, CTL, ATL, TSB)
- Familiarity with JSON

---

## Understanding the Training Process

### What is Fine-tuning?

Fine-tuning adapts a pre-trained language model to a specific task. Instead of training from scratch (which would require massive datasets and compute), we take a model that already understands language and teach it our specific domain.

```
Pre-trained Model                Fine-tuned Model
(General language)     ─────▶    (Fitness coaching)
     │                                │
     │ Knows grammar                  │ Knows grammar
     │ Knows facts                    │ Knows facts
     │ Knows reasoning                │ Knows reasoning
     │                                │ + Knows TSS/CTL/ATL
     │                                │ + Knows training principles
     │                                │ + Outputs structured JSON
```

### What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) is an efficient fine-tuning method:

1. **Quantization**: Compress model to 4-bit (reduces memory 4x)
2. **LoRA**: Only train small "adapter" layers (reduces parameters 1000x)

This allows training 7B parameter models on consumer GPUs.

### Supervised Fine-tuning (SFT) vs RLHF

| Method | How it Works | Best For |
|--------|--------------|----------|
| **SFT** | Learn from examples directly | Structured outputs, clear correct answers |
| **RLHF** | Learn from human preferences | Subjective quality, nuanced responses |

We use **SFT** because:
- Our outputs are structured JSON (clear right/wrong)
- We have specific coaching principles to follow
- It's simpler and more data-efficient

### The Training Pipeline

```
┌────────────────┐
│ 1. Collect or  │
│    Generate    │
│    Training    │──────┐
│    Data        │      │
└────────────────┘      │
                        ▼
                ┌───────────────┐
                │ 2. Format as  │
                │    JSONL with │
                │    messages   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ 3. Load Base  │
                │    Model with │
                │    QLoRA      │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ 4. Train with │
                │    SFTTrainer │
                │    (2-4 hrs)  │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ 5. Save LoRA  │
                │    Adapters   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ 6. Convert to │
                │    GGUF for   │
                │    Ollama     │
                └───────────────┘
```

---

## Preparing Training Data

### Data Format

Training data must be in JSONL format (one JSON object per line) with a `messages` field containing the conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert endurance coach..."
    },
    {
      "role": "user",
      "content": "Context: CTL=45, ATL=60, TSB=-15\nFeedback: I'm feeling tired..."
    },
    {
      "role": "assistant",
      "content": "{\"analysis\": {...}, \"modifications\": [...], ...}"
    }
  ]
}
```

### Types of Training Examples

#### 1. Plan Modification Examples
Teaching the model to analyze and modify existing plans.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert endurance coach. Analyze the athlete's status and suggest plan modifications. Output valid JSON only."
    },
    {
      "role": "user",
      "content": "ATHLETE CONTEXT:\n{\"athlete\": {\"name\": \"Sarah\", \"primary_sport\": \"rowing\"}, \"current_metrics\": {\"fitness_ctl\": 48, \"fatigue_atl\": 72, \"form_tsb\": -24, \"form_status\": \"very_fatigued\"}, \"upcoming_workouts\": [{\"id\": \"w1\", \"date\": \"2024-01-15\", \"name\": \"4x2K Intervals\", \"sport\": \"rowing\", \"estimated_tss\": 95}]}\n\nUSER FEEDBACK:\nI did a hard erg session yesterday and my legs are really tired. RPE was 9. Also feeling a bit under the weather."
    },
    {
      "role": "assistant",
      "content": "{\"analysis\": {\"current_status\": \"Athlete is in a significantly fatigued state with TSB of -24. Combined with high RPE from yesterday and feeling unwell, recovery is priority.\", \"key_concerns\": [\"Very negative TSB indicates accumulated fatigue\", \"High RPE (9) suggests incomplete recovery\", \"Possible illness onset requires caution\"], \"recommendations_summary\": \"Reduce intensity significantly for the next 3-4 days. Replace interval session with easy aerobic work.\"}, \"modifications\": [{\"workout_id\": \"w1\", \"date\": \"2024-01-15\", \"original_name\": \"4x2K Intervals\", \"action\": \"modify\", \"changes\": {\"name\": \"Easy Steady State\", \"duration_minutes\": {\"from\": 75, \"to\": 45}, \"intensity\": {\"from\": \"Zone 4-5 intervals\", \"to\": \"Zone 2 steady\"}, \"estimated_tss\": {\"from\": 95, \"to\": 35}, \"notes\": \"Easy paddle, stay in Zone 2. If feeling worse, skip entirely.\"}}], \"weekly_load_adjustment\": {\"current_weekly_tss\": 420, \"recommended_weekly_tss\": 280, \"reason\": \"Significant fatigue and possible illness require load reduction\"}, \"athlete_message\": \"I can see you're quite fatigued and not feeling well. Let's back off this week to let your body recover. I've changed tomorrow's intervals to an easy steady state session - if you're still feeling unwell, skip it entirely. Listen to your body and don't push through illness.\"}"
    }
  ]
}
```

#### 2. Weekly Plan Generation Examples
Teaching the model to create balanced weekly plans.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert endurance coach. Create a weekly training plan. Output valid JSON only."
    },
    {
      "role": "user",
      "content": "ATHLETE CONTEXT:\n{\"athlete\": {\"name\": \"Mike\", \"primary_sport\": \"cycling\"}, \"current_metrics\": {\"fitness_ctl\": 65, \"fatigue_atl\": 55, \"form_tsb\": 10, \"form_status\": \"fresh\"}}\n\nTRAINING GOALS:\nBuild towards a century ride in 6 weeks\n\nCONSTRAINTS:\nCan only ride weekday mornings (max 90 min) and weekends are open"
    },
    {
      "role": "assistant",
      "content": "{\"plan_summary\": {\"focus\": \"Endurance base building with weekend long rides\", \"total_tss\": 380, \"total_hours\": 9.5, \"key_sessions\": [\"Saturday long ride progression\", \"Tuesday tempo intervals\"]}, \"workouts\": [{\"day\": \"monday\", \"date\": \"2024-01-15\", \"name\": \"Recovery Spin\", \"sport\": \"cycling\", \"duration_minutes\": 45, \"estimated_tss\": 25, \"description\": \"Easy spinning to start the week\", \"steps\": [{\"step_type\": \"active\", \"duration_type\": \"time\", \"duration_value\": 2700, \"target_type\": \"heart_rate\", \"target_low\": 110, \"target_high\": 130, \"notes\": \"Zone 1-2, very easy\"}]}, {\"day\": \"tuesday\", \"date\": \"2024-01-16\", \"name\": \"Tempo Intervals\", \"sport\": \"cycling\", \"duration_minutes\": 75, \"estimated_tss\": 70, \"description\": \"3x10min at tempo with 5min recovery\", \"steps\": [{\"step_type\": \"warmup\", \"duration_type\": \"time\", \"duration_value\": 900, \"target_type\": \"heart_rate\", \"target_low\": 110, \"target_high\": 130, \"notes\": \"Easy spinning warmup\"}, {\"step_type\": \"active\", \"duration_type\": \"time\", \"duration_value\": 600, \"target_type\": \"power\", \"target_low\": 200, \"target_high\": 220, \"notes\": \"Tempo effort, sustainable but challenging\"}, {\"step_type\": \"recovery\", \"duration_type\": \"time\", \"duration_value\": 300, \"target_type\": \"open\", \"notes\": \"Easy spinning recovery\"}]}], \"athlete_message\": \"This week focuses on building your endurance base for the century. The long ride Saturday is your key session - aim to finish feeling like you could do more. Tuesday's tempo work will build your sustainable power. Don't neglect the easy days - they're crucial for adaptation.\"}"
    }
  ]
}
```

#### 3. General Coaching Conversations
Teaching general training knowledge and communication.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert endurance coach specializing in rowing, cycling, running, and triathlon."
    },
    {
      "role": "user",
      "content": "What's the difference between FTP and LTHR? How do they relate?"
    },
    {
      "role": "assistant",
      "content": "FTP (Functional Threshold Power) and LTHR (Lactate Threshold Heart Rate) both represent your threshold intensity - the point where lactate begins to accumulate faster than your body can clear it. This is roughly the intensity you could sustain for about an hour.\n\n**FTP** is measured in watts and represents your power output at threshold. It's objective and doesn't fluctuate day-to-day based on fatigue, hydration, or temperature.\n\n**LTHR** is measured in beats per minute and represents your heart rate at threshold. It can vary based on factors like fatigue, heat, dehydration, and caffeine.\n\n**How they relate:**\n- Both indicate the same physiological threshold\n- Training at the same percentage of each should feel similar\n- Power is more reliable for prescribing and tracking workouts\n- Heart rate is useful when power isn't available and for monitoring fatigue\n\n**Practical use:**\n- Use power zones for interval workouts (more precise)\n- Use heart rate zones for easy aerobic work\n- If HR is high at your usual power, you may be fatigued\n- If HR is low at usual power, you may be getting fitter"
    }
  ]
}
```

### Generating Synthetic Data

Use the provided script to generate training examples:

```bash
python generate_training_data.py \
  --output ./training_data.jsonl \
  --count 1000 \
  --types plan_modification weekly_plan coaching
```

The script generates realistic scenarios by:
1. Creating random athlete profiles (CTL 20-80, various sports)
2. Generating appropriate workouts for their fitness
3. Simulating various user feedback scenarios
4. Creating appropriate AI responses

### Data Quality Guidelines

**Good Examples:**
- Consistent with training science principles
- Appropriate load adjustments for the situation
- Clear reasoning in analysis
- Valid JSON that passes schema validation

**Bad Examples:**
- Suggesting hard training when TSB is very negative
- Ignoring injury/illness mentions
- Invalid JSON or missing fields
- Unrealistic TSS values

### Validating Your Data

```bash
python validate_data.py ./training_data.jsonl
```

This checks:
- JSON validity
- Required fields present
- Values within reasonable ranges
- Consistent coaching principles

---

## Training on Google Colab

### Step 1: Open the Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `train_model.ipynb` or open from GitHub
3. Change runtime to GPU (Runtime → Change runtime type → T4/A100)

### Step 2: Install Dependencies

```python
!pip install -q transformers datasets peft bitsandbytes trl accelerate
```

### Step 3: Upload Training Data

```python
from google.colab import files
uploaded = files.upload()  # Upload training_data.jsonl
```

### Step 4: Load Base Model with QLoRA

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,                      # Rank of adaptation matrices
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 3,765,948,416 || trainable%: 0.362
```

### Step 5: Prepare Dataset

```python
from datasets import load_dataset

def format_example(example):
    """Format messages into training format."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
dataset = dataset.map(format_example)
dataset = dataset.train_test_split(test_size=0.1)
```

### Step 6: Configure Training

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./fitness-coach-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    bf16=True,
    max_seq_length=2048,
    packing=True,  # Pack multiple examples into single sequence
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
)
```

### Step 7: Train

```python
trainer.train()
```

Expected output:
```
Step 100/750: loss=1.234
Step 200/750: loss=0.987
Step 300/750: loss=0.765
...
Training completed in 2h 34m
```

### Step 8: Save and Download

```python
# Save LoRA adapters
trainer.save_model("./fitness-coach-lora")

# Merge with base model (optional, for easier deployment)
from peft import PeftModel
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./fitness-coach-merged")
tokenizer.save_pretrained("./fitness-coach-merged")

# Download
from google.colab import files
!zip -r fitness-coach-lora.zip ./fitness-coach-lora
files.download("fitness-coach-lora.zip")
```

---

## Training Locally (Optional)

If you have a GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100):

```bash
# Install dependencies
pip install transformers datasets peft bitsandbytes trl accelerate

# Run training script
python -c "
from train_model import train
train(
    data_path='./training_data.jsonl',
    output_dir='./fitness-coach-lora',
    num_epochs=3,
)
"
```

---

## Creating the Ollama Model

### Step 1: Convert to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert model to GGUF
python convert.py ../fitness-coach-merged --outtype f16 --outfile fitness-coach.gguf

# Quantize for efficiency
./quantize fitness-coach.gguf fitness-coach-q4.gguf q4_K_M
```

### Step 2: Create Modelfile

Create `Modelfile`:

```dockerfile
FROM ./fitness-coach-q4.gguf

SYSTEM """You are an expert endurance sports coach specializing in rowing, cycling, running, and triathlon. You provide evidence-based, personalized training advice.

When asked to analyze or modify training plans, you MUST output valid JSON following this structure:
{
  "analysis": { "current_status": "...", "key_concerns": [...], "recommendations_summary": "..." },
  "modifications": [ { "workout_id": "...", "action": "modify|skip|replace", ... } ],
  "new_workouts": [ { "date": "...", "name": "...", ... } ],
  "athlete_message": "..."
}

Always consider:
- Current form (TSB): negative = fatigued, positive = fresh
- Training principles: progressive overload, recovery, specificity
- Injury and illness require reduced load
- Output ONLY valid JSON when asked for plan modifications
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 2048
```

### Step 3: Create Model

```bash
ollama create fitness-coach -f Modelfile
```

### Step 4: Test

```bash
ollama run fitness-coach "My CTL is 45, ATL is 60, TSB is -15. I'm feeling tired. What should I do?"
```

---

## Testing and Evaluation

### Manual Testing

Test various scenarios:

```python
scenarios = [
    # Recovery needed
    {"ctl": 50, "atl": 70, "tsb": -20, "feedback": "Very tired, legs heavy"},

    # Fresh and ready
    {"ctl": 60, "atl": 50, "tsb": 10, "feedback": "Feeling great, ready to push"},

    # Illness
    {"ctl": 55, "atl": 55, "tsb": 0, "feedback": "Feeling sick, sore throat"},

    # Injury
    {"ctl": 45, "atl": 50, "tsb": -5, "feedback": "Knee pain after running"},
]
```

### Evaluation Metrics

1. **JSON Validity**: Does output parse as valid JSON?
2. **Schema Compliance**: Does it follow our expected schema?
3. **Coaching Principles**: Are recommendations appropriate?
4. **User Preference**: Do test users find advice helpful?

### A/B Testing

Compare fine-tuned model vs base model:
- Same context, different models
- Measure JSON validity rate
- Rate coaching quality (blind evaluation)

---

## Deployment

### 1. Start Ollama Server

```bash
ollama serve
```

### 2. Configure Backend

In `.env`:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=fitness-coach
```

### 3. Run Backend

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. Test Integration

```bash
curl -X POST http://localhost:8000/api/v1/ai/plan/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feedback": "Feeling tired today", "days_forward": 7}'
```

---

## Iterating and Improving

### When to Retrain

1. Model gives inappropriate advice consistently
2. New training scenarios added (new sport, new metrics)
3. Users report unhelpful responses

### Collecting Real Feedback

Log user interactions:
- Accepted modifications (positive examples)
- Refined modifications (shows what was wrong)
- Rejected modifications (negative examples)

### Continuous Improvement Cycle

```
┌─────────────────┐
│ Deploy Model    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Collect Usage   │
│ Logs            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Identify Issues │
│ & Successes     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create New      │
│ Training Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrain Model   │
└────────┬────────┘
         │
         └──────────▶ (repeat)
```

---

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use smaller model (Gemma 2B)

### Training Loss Not Decreasing

- Check data quality
- Increase learning rate
- Ensure examples are properly formatted

### Model Outputs Invalid JSON

- Add more JSON examples to training
- Strengthen system prompt
- Use output validation and retry

### Ollama Model Not Loading

- Check GGUF conversion
- Verify Modelfile path
- Check available memory
