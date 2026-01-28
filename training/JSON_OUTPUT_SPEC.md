# AI Coach JSON Output Specification

This document defines the exact JSON schema that the AI coach model should output for plan modifications.

## Overview

The model outputs structured JSON to enable:
1. Programmatic parsing and validation
2. Preview generation for users
3. Safe application to the database

**Important**: The model must output ONLY valid JSON - no prose before or after.

---

## Plan Modification Response

Used when analyzing existing plans and suggesting modifications.

### Full Schema

```json
{
  "analysis": {
    "current_status": "string - Brief assessment of athlete's current state",
    "key_concerns": ["string - List of concerns from metrics and feedback"],
    "recommendations_summary": "string - High-level summary of changes"
  },
  "modifications": [
    {
      "workout_id": "string - ID from upcoming_workouts (REQUIRED)",
      "date": "string - YYYY-MM-DD format (REQUIRED)",
      "original_name": "string - Name of original workout",
      "action": "string - modify|skip|replace (REQUIRED)",
      "changes": {
        "name": "string - New name (if changing)",
        "duration_minutes": {
          "from": "number - Original duration",
          "to": "number - New duration (REQUIRED)"
        },
        "intensity": {
          "from": "string - Original intensity description",
          "to": "string - New intensity description (REQUIRED)"
        },
        "estimated_tss": {
          "from": "number - Original TSS",
          "to": "number - New TSS (REQUIRED)"
        },
        "notes": "string - Explanation for this change"
      }
    }
  ],
  "new_workouts": [
    {
      "date": "string - YYYY-MM-DD format (REQUIRED)",
      "name": "string - Workout name (REQUIRED)",
      "sport": "string - rowing|cycling|running|swimming|strength|other (REQUIRED)",
      "duration_minutes": "number - Duration in minutes (REQUIRED)",
      "estimated_tss": "number - Expected TSS",
      "description": "string - Workout description",
      "steps": [
        {
          "step_type": "string - warmup|active|recovery|cooldown|rest (REQUIRED)",
          "duration_type": "string - time|distance|calories|open (REQUIRED)",
          "duration_value": "number - Seconds for time, meters for distance",
          "target_type": "string - heart_rate|power|pace|cadence|open",
          "target_low": "number - Lower bound of target",
          "target_high": "number - Upper bound of target",
          "notes": "string - Step-specific notes"
        }
      ]
    }
  ],
  "weekly_load_adjustment": {
    "current_weekly_tss": "number - Current planned weekly TSS",
    "recommended_weekly_tss": "number - Recommended weekly TSS",
    "reason": "string - Explanation for load change"
  },
  "athlete_message": "string - Friendly message explaining changes (REQUIRED)"
}
```

### Required vs Optional Fields

| Field | Required | Notes |
|-------|----------|-------|
| `analysis` | Optional | But recommended for transparency |
| `modifications` | Required* | Array, can be empty |
| `new_workouts` | Required* | Array, can be empty |
| `weekly_load_adjustment` | Optional | Include when suggesting load changes |
| `athlete_message` | **Required** | Always include to communicate with user |

*At least one of `modifications` or `new_workouts` should have content.

### Actions

| Action | Description | Behavior |
|--------|-------------|----------|
| `modify` | Change existing workout | Apply `changes` to workout |
| `skip` | Remove workout | Delete from plan |
| `replace` | Replace entirely | Apply all `changes`, essentially new workout |

### Example: Fatigue Response

```json
{
  "analysis": {
    "current_status": "Athlete has TSB of -18 indicating significant fatigue. Recent 90 TSS session with RPE 9 suggests incomplete recovery.",
    "key_concerns": [
      "TSB well below -15 threshold",
      "High RPE indicates perception of hard effort",
      "Tomorrow's planned intervals will add more fatigue"
    ],
    "recommendations_summary": "Reduce tomorrow's intensity and duration. Consider reducing weekly load by 20%."
  },
  "modifications": [
    {
      "workout_id": "abc123",
      "date": "2024-01-15",
      "original_name": "5x1K Intervals",
      "action": "modify",
      "changes": {
        "name": "Easy Steady State",
        "duration_minutes": {"from": 75, "to": 50},
        "intensity": {"from": "Zone 4-5 intervals", "to": "Zone 2 steady"},
        "estimated_tss": {"from": 85, "to": 40},
        "notes": "Replace hard intervals with easy aerobic work to aid recovery"
      }
    }
  ],
  "new_workouts": [],
  "weekly_load_adjustment": {
    "current_weekly_tss": 420,
    "recommended_weekly_tss": 340,
    "reason": "Reduce load to allow recovery from accumulated fatigue"
  },
  "athlete_message": "I can see you're carrying quite a bit of fatigue. Let's scale back tomorrow's session to easy aerobic work instead of intervals. This will help you recover while maintaining your fitness. We can get back to harder work once your form improves."
}
```

### Example: Skip Workout

```json
{
  "analysis": {
    "current_status": "Athlete reports illness symptoms (sore throat, fatigue). Training during illness can prolong recovery.",
    "key_concerns": [
      "Illness onset requires rest",
      "Training while sick can worsen condition",
      "Better to lose 3 days than 3 weeks"
    ],
    "recommendations_summary": "Skip planned workouts until symptoms resolve."
  },
  "modifications": [
    {
      "workout_id": "def456",
      "date": "2024-01-16",
      "original_name": "Tempo Run",
      "action": "skip",
      "changes": {
        "notes": "Skip due to illness - rest is more important"
      }
    },
    {
      "workout_id": "ghi789",
      "date": "2024-01-17",
      "original_name": "Long Ride",
      "action": "skip",
      "changes": {
        "notes": "Skip until feeling better"
      }
    }
  ],
  "new_workouts": [],
  "athlete_message": "When you're feeling unwell, rest is the best medicine. I've cleared your schedule for the next couple of days. Focus on hydration, sleep, and letting your body recover. We can reassess once you're feeling better - don't rush back."
}
```

---

## Weekly Plan Response

Used when generating a new weekly training plan.

### Schema

```json
{
  "plan_summary": {
    "focus": "string - Main focus of this training week",
    "total_tss": "number - Total TSS for the week",
    "total_hours": "number - Total training hours",
    "key_sessions": ["string - List of most important sessions"]
  },
  "workouts": [
    {
      "day": "string - monday|tuesday|wednesday|thursday|friday|saturday|sunday (REQUIRED)",
      "date": "string - YYYY-MM-DD format (REQUIRED)",
      "name": "string - Workout name (REQUIRED)",
      "sport": "string - rowing|cycling|running|swimming|strength|rest (REQUIRED)",
      "duration_minutes": "number - Duration in minutes (REQUIRED)",
      "estimated_tss": "number - Expected TSS",
      "description": "string - Workout description",
      "steps": [
        {
          "step_type": "string - warmup|active|recovery|cooldown|rest (REQUIRED)",
          "duration_type": "string - time|distance|calories|open (REQUIRED)",
          "duration_value": "number - Seconds for time, meters for distance",
          "target_type": "string - heart_rate|power|pace|cadence|open",
          "target_low": "number - Lower bound of target",
          "target_high": "number - Upper bound of target",
          "notes": "string - Step-specific notes"
        }
      ]
    }
  ],
  "athlete_message": "string - Message explaining the plan (REQUIRED)"
}
```

### Example: Base Building Week

```json
{
  "plan_summary": {
    "focus": "Aerobic base building with progressive long ride",
    "total_tss": 320,
    "total_hours": 7.5,
    "key_sessions": [
      "Saturday long ride (2.5 hours)",
      "Wednesday threshold intervals"
    ]
  },
  "workouts": [
    {
      "day": "monday",
      "date": "2024-01-15",
      "name": "Rest Day",
      "sport": "rest",
      "duration_minutes": 0,
      "estimated_tss": 0,
      "description": "Complete rest or light stretching"
    },
    {
      "day": "tuesday",
      "date": "2024-01-16",
      "name": "Easy Spin",
      "sport": "cycling",
      "duration_minutes": 60,
      "estimated_tss": 40,
      "description": "Easy aerobic ride",
      "steps": [
        {
          "step_type": "active",
          "duration_type": "time",
          "duration_value": 3600,
          "target_type": "heart_rate",
          "target_low": 110,
          "target_high": 140,
          "notes": "Zone 2, conversational pace"
        }
      ]
    },
    {
      "day": "wednesday",
      "date": "2024-01-17",
      "name": "Threshold Intervals",
      "sport": "cycling",
      "duration_minutes": 75,
      "estimated_tss": 75,
      "description": "2x15min at threshold with 10min recovery",
      "steps": [
        {
          "step_type": "warmup",
          "duration_type": "time",
          "duration_value": 900,
          "target_type": "heart_rate",
          "target_low": 110,
          "target_high": 130,
          "notes": "Easy warmup"
        },
        {
          "step_type": "active",
          "duration_type": "time",
          "duration_value": 900,
          "target_type": "power",
          "target_low": 230,
          "target_high": 250,
          "notes": "First threshold interval"
        },
        {
          "step_type": "recovery",
          "duration_type": "time",
          "duration_value": 600,
          "target_type": "open",
          "notes": "Easy spinning"
        },
        {
          "step_type": "active",
          "duration_type": "time",
          "duration_value": 900,
          "target_type": "power",
          "target_low": 230,
          "target_high": 250,
          "notes": "Second threshold interval"
        },
        {
          "step_type": "cooldown",
          "duration_type": "time",
          "duration_value": 600,
          "target_type": "open",
          "notes": "Easy cool down"
        }
      ]
    },
    {
      "day": "saturday",
      "date": "2024-01-20",
      "name": "Long Endurance Ride",
      "sport": "cycling",
      "duration_minutes": 150,
      "estimated_tss": 120,
      "description": "Long aerobic ride, mostly Zone 2",
      "steps": [
        {
          "step_type": "active",
          "duration_type": "time",
          "duration_value": 9000,
          "target_type": "heart_rate",
          "target_low": 120,
          "target_high": 145,
          "notes": "Steady Zone 2, can include brief Zone 3 efforts on hills"
        }
      ]
    }
  ],
  "athlete_message": "This week focuses on building your aerobic engine. The Saturday long ride is your key session - aim to finish feeling like you could keep going. Wednesday's threshold work keeps your top-end sharp without digging too deep. Keep Tuesday and Thursday truly easy to absorb the training."
}
```

---

## Validation Rules

### TSS Bounds
- Minimum: 0
- Maximum: 500 (single workout)
- Typical ranges:
  - Recovery: 20-40
  - Easy endurance: 40-70
  - Moderate: 70-100
  - Hard: 100-150
  - Very hard: 150+

### Duration Bounds
- Minimum: 5 minutes
- Maximum: 480 minutes (8 hours)

### Valid Sports
- `rowing`
- `cycling`
- `running`
- `swimming`
- `strength`
- `other`
- `rest`

### Valid Step Types
- `warmup`
- `active`
- `recovery`
- `cooldown`
- `rest`

### Valid Duration Types
- `time` (value in seconds)
- `distance` (value in meters)
- `calories` (value in calories)
- `open` (no target duration)

### Valid Target Types
- `heart_rate` (value in bpm)
- `power` (value in watts)
- `pace` (value in seconds per km)
- `cadence` (value in rpm)
- `open` (no target)

---

## Error Handling

If the model cannot generate valid output, it should still return JSON:

```json
{
  "analysis": {
    "current_status": "Unable to fully analyze due to missing information",
    "key_concerns": ["Insufficient context provided"],
    "recommendations_summary": "More information needed"
  },
  "modifications": [],
  "new_workouts": [],
  "athlete_message": "I need more information to provide specific recommendations. Could you tell me more about your recent training and how you're feeling?"
}
```

---

## Common Mistakes to Avoid

### 1. Prose Around JSON
**Wrong:**
```
Here are my suggestions:
{"modifications": [...]}
I hope this helps!
```

**Correct:**
```json
{"modifications": [...], "athlete_message": "I hope this helps!"}
```

### 2. Missing Required Fields
**Wrong:**
```json
{
  "modifications": [{"workout_id": "abc", "action": "modify"}]
}
```

**Correct:**
```json
{
  "modifications": [{"workout_id": "abc", "date": "2024-01-15", "action": "modify", "changes": {...}}],
  "athlete_message": "..."
}
```

### 3. Invalid Action Values
**Wrong:**
```json
{"action": "change"}
```

**Correct:**
```json
{"action": "modify"}
```

### 4. Duration Without From/To Structure
**Wrong:**
```json
{"duration_minutes": 60}
```

**Correct:**
```json
{"duration_minutes": {"from": 90, "to": 60}}
```
