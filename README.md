# Job Scam Detection Environment

An OpenEnv-compatible RL environment for training and evaluating agents on job-scam detection.

Each episode presents a real-world job opportunity (WhatsApp message, email, Telegram post, or job board post). The agent investigates by requesting hidden context fields and then submits a classification label. All grading and reward computation is fully programmatic — no LLM is involved in the environment itself.

---

## Quick start

```bash
# Install dependencies
uv sync

# Start the environment server
uv run server

Or with Docker:

```bash
docker build -t job_scam_env-env:latest .
python inference.py   # connects via ENV_URL (HF Space)
```
## Task Structure

The environment supports **three difficulty levels**:

| Task | Description |
|---|---|
| **Easy** | Single-message classification (no investigation) |
| **Medium** | Multi-step investigation using context fields |
| **Hard** | Advanced multi-signal reasoning (in development) |

---

## Episode structure

### Easy task

```

reset()
└─ Observation: query_type, initial_query, step_budget

step(classify(label)) exactly once, terminal
└─ Observation: predicted_label, actual_label
Reward: classification_reward
done: true

```

Maximum steps per episode: **1**

No context requests are allowed.

---

### Medium task

```
reset()
  └─ Observation: query_type, initial_query, available_context, step_budget

step(request_<field>)           × 0–4 times
  └─ Observation: field_content, step_budget
     Reward: signal_reward | redundancy_penalty | irrelevant_field_penalty

step(classify(label))           exactly once, terminal
  └─ Observation: predicted_label, actual_label, step_budget
     Reward: classification_reward + total_steps_taken_reward
     done: true

── OR ──

step budget exhausted without classify → timeout
  └─ done: true
```

Maximum steps per episode: **5** (4 info requests + 1 classify, or fewer).

---

## Action space

### Easy task

| Action | Description |
|---|---|
| `classify(label)` | Submit final verdict — **terminates the episode** |

Valid labels: `legit` · `scam`

---

### Medium task

| Action | Description |
|---|---|
| `request_recruiter_profile` | Fetch recruiter name, contact, and bio |
| `request_company_profile` | Fetch company domain, hiring policy, anti-scam statements |
| `request_thread_history` | Fetch prior message thread between sender and candidate |
| `request_job_post_comments` | Fetch public comments or forwarded warnings |
| `classify(label)` | Submit final verdict — **terminates the episode** |

Valid labels: `legit` · `suspicious` · `scam` · `insufficient_info`

---

## Reward structure

### Easy task

Classification-only reward:

| Condition | Reward |
|---|---|
| Correct classification | `+1.0` |
| Incorrect classification (scam → legit) | `0.1` |
| Incorrect classification (legit → scam) | `0.0` |

No intermediate rewards. No step penalties.

### Medium task

#### Per-step information reward

```
signal_score(field) = (|red_categories| + |green_categories|) /
                      total_unique_categories_in_sample
```

| Condition | Reward |
|---|---|
| Valid, non-redundant field (signal > 0) | `+0.10 × signal_score` |
| Field already requested this episode | `−0.20` |
| Field has zero signal (empty / N/A) | `−0.05` |

---

### Terminal classification reward

```
classification_reward   = REWARD_MATRIX[predicted][ground_truth]
alpha                   = +0.1  if correct  else  −0.1
total_steps_taken_reward = alpha × remaining_steps_at_classification
terminal_reward          = classification_reward + total_steps_taken_reward
```

Classification reward matrix (`REWARD_MATRIX[predicted][gt]`):

|  | gt: legit | gt: suspicious | gt: scam | gt: insuf |
|---|---|---|---|---|
| **pred: legit** | +1.00 | −0.30 | −1.00 | −0.20 |
| **pred: suspicious** | −0.10 | +1.00 | −0.30 | −0.10 |
| **pred: scam** | −0.50 | −0.10 | +1.00 | −0.30 |
| **pred: insuf** | −0.20 | −0.20 | −0.50 | +1.00 |

The asymmetric penalties reflect real-world stakes: calling a scam `legit` is maximally penalised at −1.00.

---

### Timeout

If the agent exhausts all 5 steps without classifying, there will be some timeout penalty.

---

## Observation contract

The client never receives: ground truth label, field signal scores, red/green flag categories, or internal reward equations.

---

## Dataset

### Easy dataset

Stored in:
```
server/data_task_easy.jsonl
```
Each sample contains:

- `sample_id`
- `query_type`
- `initial_query`
- `ground_truth` (`legit` | `scam`)

---

### Medium dataset

Four built-in samples cover all four query types and all four GT labels:

| sample_id | query_type | ground_truth |
|---|---|---|
| job_001 | job_post | legit |
| job_002 | email | suspicious |
| job_003 | whatsapp_msg | scam |
| job_004 | telegram_msg | insufficient_info |

Stored in:
```
server/data_task_medium.jsonl
```

Each sample contains:
- `red_flag_categories`
- `green_flag_categories`

These are used internally to compute signal scores and are never exposed to the agent.

To extend the dataset, add entries to the `_DATASET` list in:
```
server/job_scam_env_environment.py
```

---

## Environment variables (inference)

| Variable | Description |
|---|---|
| `API_KEY/HF_TOKEN` | Authentication key (required by validator) |
| `API_BASE_URL` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Model identifier |
| `ENV_URL` | URL of deployed Hugging Face Space |
| `LOCAL_IMAGE_NAME` | Local Docker image name (for development only) |
| `PORT` | Port used by the environment server (default: 8000) |