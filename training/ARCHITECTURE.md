# AI Coach System Architecture

This document describes the hybrid architecture for the AI fitness coach, explaining how the fine-tuned LLM integrates with the application.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │  Feedback Input │  │ Suggested Changes│  │  Accept / Refine / Reject  │  │
│  │  "I'm feeling   │  │  [Modification]  │  │                            │  │
│  │   tired..."     │  │  [Modification]  │  │  [Apply] [Refine] [Cancel] │  │
│  └────────┬────────┘  └────────▲─────────┘  └─────────────┬──────────────┘  │
│           │                    │                          │                  │
└───────────┼────────────────────┼──────────────────────────┼──────────────────┘
            │                    │                          │
            ▼                    │                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         CONTEXT BUILDER                                 │  │
│  │                                                                         │  │
│  │   Input: User ID, Feedback                                              │  │
│  │                                                                         │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │   │ Fetch User  │  │ Get Recent  │  │Get Upcoming │  │  Calculate  │   │  │
│  │   │  Profile &  │──│ Activities  │──│  Planned    │──│ CTL/ATL/TSB │   │  │
│  │   │ Thresholds  │  │ (7 days)    │  │  Workouts   │  │             │   │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │                                                                         │  │
│  │   Output: Structured JSON context + System prompt                       │  │
│  └────────────────────────────────────────┬────────────────────────────────┘  │
│                                           │                                   │
│                                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           OLLAMA (LLM)                                  │  │
│  │                                                                         │  │
│  │   Model: fitness-coach (fine-tuned Mistral 7B)                         │  │
│  │                                                                         │  │
│  │   Input:  System prompt with athlete context                           │  │
│  │           + User feedback                                              │  │
│  │           + (Optional) Previous suggestions for refinement              │  │
│  │                                                                         │  │
│  │   Output: Structured JSON with modifications                            │  │
│  │                                                                         │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │   │ {                                                               │   │  │
│  │   │   "analysis": {...},                                           │   │  │
│  │   │   "modifications": [...],                                       │   │  │
│  │   │   "new_workouts": [...],                                        │   │  │
│  │   │   "athlete_message": "..."                                      │   │  │
│  │   │ }                                                               │   │  │
│  │   └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────┬────────────────────────────────┘  │
│                                           │                                   │
│                                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        WORKOUT MODIFIER                                 │  │
│  │                                                                         │  │
│  │   1. Extract JSON from LLM response                                    │  │
│  │   2. Validate with Pydantic schemas                                    │  │
│  │   3. Apply sanity checks (TSS limits, duration bounds, etc.)           │  │
│  │   4. Generate preview for user                                          │  │
│  │   5. (On approval) Apply changes to MongoDB                            │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Context Builder (`context_builder.py`)

**Purpose**: Gather all relevant data about the athlete and format it for the LLM.

**Data Sources**:
- User profile (name, primary sport, thresholds)
- Current metrics (CTL, ATL, TSB calculated from activities)
- Recent activities (last 7 days)
- Upcoming planned workouts (next 14 days)

**Output Format**:
```json
{
  "athlete": {
    "name": "John",
    "primary_sport": "rowing",
    "thresholds": {
      "lthr_bpm": 170,
      "ftp_watts": 250
    }
  },
  "current_metrics": {
    "fitness_ctl": 52.3,
    "fatigue_atl": 68.1,
    "form_tsb": -15.8,
    "form_status": "fatigued",
    "form_description": "Building fitness, manage recovery carefully"
  },
  "recent_activities": [...],
  "upcoming_workouts": [...]
}
```

### 2. LLM (Ollama)

**Purpose**: Make coaching decisions based on context.

**Input**:
- System prompt with structured context
- User's feedback about how they're feeling
- (Optional) Previous suggestions for refinement

**Training Focus**:
- Understanding training metrics (CTL, ATL, TSB)
- Making appropriate load adjustments
- Recognizing injury/illness signals
- Sport-specific knowledge
- Structured JSON output

**Output**: JSON following the schema in `JSON_OUTPUT_SPEC.md`

### 3. Workout Modifier (`workout_modifier.py`)

**Purpose**: Validate and apply LLM suggestions.

**Validation Steps**:
1. Extract JSON (handle markdown code blocks)
2. Parse with Pydantic models
3. Sanity checks:
   - TSS: 0-500 range
   - Duration: 5-480 minutes
   - Valid sport types
   - Valid step types
4. Generate human-readable preview
5. Apply to database on user approval

**Why Validation Matters**:
- LLMs can hallucinate invalid values
- Prevents data corruption
- Allows user oversight before changes

### 4. FIT Generator (`fit_generator.py`)

**Purpose**: Generate downloadable FIT files for Garmin devices.

**Process**:
1. Read planned workout from database
2. Convert steps to FIT workout protocol
3. Generate binary FIT file with proper CRC
4. Return for download

## Data Flow

### Plan Analysis Flow

```
1. User enters feedback: "Feeling tired, RPE 8 yesterday"
                    │
                    ▼
2. Context Builder fetches:
   - User profile & thresholds
   - Last 7 days of activities
   - Next 14 days of workouts
   - Current CTL=52, ATL=68, TSB=-16
                    │
                    ▼
3. Build system prompt with all context
                    │
                    ▼
4. Send to Ollama (fitness-coach model)
                    │
                    ▼
5. LLM responds with JSON:
   {
     "modifications": [
       {"workout_id": "abc", "action": "modify", "changes": {...}}
     ],
     "athlete_message": "Given your high fatigue..."
   }
                    │
                    ▼
6. Workout Modifier validates JSON
                    │
                    ▼
7. Generate preview for user
                    │
                    ▼
8. User reviews and either:
   - Accepts → Changes applied to DB
   - Refines → Feedback sent back to LLM (step 4)
   - Rejects → No changes made
```

### Iterative Refinement Flow

```
User: "I'm tired"
         │
         ▼
AI suggests: "Skip Thursday's intervals"
         │
         ▼
User: "I don't want to skip it, just make it easier"
         │
         ▼
AI suggests: "Change Thursday from Z4 intervals to Z2 steady state"
         │
         ▼
User: "That works" → [Apply]
```

## Why Hybrid Architecture?

### Benefits

1. **Focused Model**: LLM only handles coaching decisions, not data manipulation
2. **Validation Layer**: Python code validates outputs before database changes
3. **User Control**: Preview before applying any changes
4. **Iterative Refinement**: Users can request adjustments without restarting
5. **Deterministic Actions**: Database operations are handled by reliable code
6. **Easy Updates**: Can improve prompts without retraining

### Comparison to End-to-End LLM

| Approach | Pros | Cons |
|----------|------|------|
| **Hybrid** | Validated outputs, user control, focused training | More code to maintain |
| **End-to-End** | Simpler architecture | Risk of invalid data, no validation |

## Error Handling

### LLM Errors

1. **Invalid JSON**: Extract JSON with regex, attempt repair
2. **Missing Fields**: Use Pydantic defaults where possible
3. **Invalid Values**: Flag in warnings, skip that modification
4. **Connection Failed**: Return helpful error message

### Validation Errors

```python
# Example validation
def validate_tss(tss: float) -> tuple[bool, Optional[str]]:
    if tss < 0:
        return False, "TSS cannot be negative"
    if tss > 500:
        return False, f"TSS of {tss} is unrealistically high"
    return True, None
```

## Performance Considerations

### Latency

- Context building: ~50-100ms (database queries)
- LLM inference: ~2-10 seconds (depending on output length)
- Validation: ~10ms
- Database updates: ~50-100ms

### Optimization Strategies

1. **Parallel Queries**: Fetch activities and workouts concurrently
2. **Caching**: Cache user thresholds and recent context
3. **Streaming**: Stream LLM response for better UX
4. **Batch Updates**: Apply all modifications in single transaction

## Security Considerations

1. **Input Sanitization**: User feedback is included in prompts
2. **Output Validation**: Never trust LLM output directly
3. **User Authorization**: Verify workout ownership before modification
4. **Rate Limiting**: Prevent abuse of expensive LLM calls
