# LLM-Guided Counterfactual Intervention for Comorbid Chronic Disease Risk in Health Insurance

This repository contains the full reproducibility code for the paper:

> **"LLM-Guided Counterfactual Intervention for Comorbid Chronic Disease Risk in Health Insurance"**  
> Anonymous Authors  
> Anonymous Institution(s)

---

## Overview

We propose a three-stage hybrid framework that combines:

1. **Stage 1 — XGBoost Multi-Class Risk Model**: Gender-stratified 4-class risk classification (Normal / HTN-only / DM-only / Comorbid) trained on KNHANES 2020–2024.
2. **Stage 2 — GPT-4o-mini Clinical Guardrail Agent**: Dynamically derives patient-specific `permitted_range` constraints in JSON, enforcing physical consistency, intervention conflict prevention, and nutritional safety floors.
3. **Stage 3 — DiCE Counterfactual Generator**: Genetic-algorithm-based counterfactual search within guardrail-constrained space, with a stepwise fallback mechanism (Class 0 → 1 → 2).

---

## Repository Structure

```
llm-cf-comorbid/
├── llm_cf_comorbid.ipynb       # Full pipeline (data → model → guardrail → DiCE)
├── experiment_results.json     # Output: guardrail CFs + no-guardrail baseline
├── knhanes_comorbid_2020_2024.csv  # Preprocessed dataset (generated on first run)
├── model_male.pkl              # Trained XGBoost model (male)
├── model_female.pkl            # Trained XGBoost model (female)
├── agent_config.pkl            # Feature lists and hyperparameters
├── df_final.pkl                # Numeric dataset for DiCE
└── README.md
```

---

## Requirements

```
Python 3.10+
xgboost==2.0.3
dice-ml==0.11
openai>=1.0.0
optuna
pyreadstat
scikit-learn
pandas
numpy
joblib
```

Install all dependencies:

```bash
pip install xgboost==2.0.3 dice-ml==0.11 openai optuna pyreadstat scikit-learn pandas numpy joblib
```

---

## Data

This study uses the **Korea National Health and Nutrition Examination Survey (KNHANES)** 2020–2024, administered by the Korea Disease Control and Prevention Agency (KDCA).

The raw SAS files (`hn20_all.sas7bdat` ~ `hn24_all.sas7bdat`) must be placed in the same directory as the notebook. Data are available at: https://knhanes.kdca.go.kr

> **Note**: Due to data use agreements, raw KNHANES files are not included in this repository. The preprocessed CSV (`knhanes_comorbid_2020_2024.csv`) is generated automatically on the first run.

---

## How to Run

### 1. Set your OpenAI API key

In `llm_cf_comorbid.ipynb`, Cell 0:

```python
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

### 2. Run all cells in order

The notebook is self-contained and executes the full pipeline sequentially:

| Section | Description |
|---|---|
| 0 | Library imports and settings |
| 1 | KNHANES 2020–2024 data merge and preprocessing |
| 2 | Feature definition and numeric dataset construction |
| 3 | XGBoost gender-stratified training (Optuna hyperparameter search) |
| 4 | GPT-4o-mini clinical guardrail agent definition |
| 5 | DiCE counterfactual generation with stepwise fallback |
| 6 | Full pipeline execution (male + female) |
| 7 | No-guardrail baseline (Table 6 comparison) |
| 8 | Export results to `experiment_results.json` |

---

## Key Design Decisions

### Non-modifiable variables (fixed in DiCE search)
The following variables are held fixed at the patient's current value and excluded from `features_to_vary`:

```python
FIXED_FEATURES = [
    'StressLevel', 'StressAwarenessRate',
    'PersonalIncomeQuartile', 'HouseholdIncomeQuartile',
    'EducationLevel', 'HealthScreeningStatus',
]
```

### Physical correction layer (guardrail hard overrides)
After the LLM generates `permitted_range`, the following rules are applied unconditionally in code:

| Rule | Description |
|---|---|
| A | BMI, WaistCirc, Weight: `[current × 0.85, current]` — coupled reduction |
| B | Energy_kcal: floor at `max(current × 0.70, 500)`, ceiling at current |
| C | Sodium_mg: floor at `max(current × 0.60, 800)`, ceiling at `current × 0.90` |
| D | Carb_g, Sugar_g: ceiling capped at current (conflict prevention) |
| E | Protein_g ≥ 60% / Potassium_mg ≥ 50% / Carb_g ≥ 20% / Fiber_g ≥ 30% of current |

### Post-processing
- Categorical variables rounded to nearest valid integer from dataset
- Anthropometric coupling enforced: the variable with the largest reduction sets the target ratio; others are scaled to match

### Stepwise fallback
When Class 0 (Normal) is unreachable within guardrail-constrained space:
```
Class 3 → Class 0 (primary)
       → Class 1 / HTN-only (fallback 1)
       → Class 2 / DM-only  (fallback 2)
```
The fallback depth is recorded in output and serves as a clinical staging indicator.

---

## Output Format

`experiment_results.json` contains two sections:

- **`results`**: Guardrail-constrained counterfactuals (4 pathways per patient)
- **`baseline_no_guardrail`**: Unconstrained DiCE output on the same patient (reproduces Table 6 hallucination examples)

---

## On-Premises Configuration

For deployments where sensitive policyholder data cannot leave the insurer's infrastructure, the guardrail agent can be replaced with a local LLM:

```
Model      : Qwen2.5-32B-Instruct
Quantization: AWQ (activation-aware weight quantization)
Serving    : vLLM 0.4.2 with PagedAttention
Context    : YaRN context extension
```

Replace the `client.chat.completions.create(...)` call in `get_clinical_guardrails()` with the vLLM endpoint accordingly.

---

## Experimental Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5-14500 (14 cores, up to 4.32 GHz) |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| OS | Windows 10 / Ubuntu 20.04+ |
| Python | 3.10+ |

---

## Citation

---

## License

This code is released for academic reproducibility purposes.  
KNHANES data use is subject to KDCA data use agreements.
