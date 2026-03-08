# LLM Agent-Driven Counterfactual Explanations for Comorbid Chronic Disease Risk Management
### An XAI Framework with Clinical Guardrails

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![vLLM](https://img.shields.io/badge/vLLM-0.4.2-purple.svg)](https://github.com/vllm-project/vllm)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![DiCE](https://img.shields.io/badge/DiCE--ML-0.11-green.svg)](https://github.com/interpretml/DiCE)

> Official implementation of the paper:  
> **"LLM Agent-Driven Counterfactual Explanations for Comorbid Chronic Disease Risk Management: An XAI Framework with Clinical Guardrails"**  
> *[Journal Name], 2025*

---

## Overview

This repository provides the full implementation of a hybrid intelligent agent pipeline for comorbid chronic disease (hypertension + diabetes) risk management. The framework integrates three core components:

1. **XGBoost** — Gender-specific multi-class risk prediction (Class 0–3)
2. **vLLM Clinical Guardrail Agent** — On-premises Qwen2.5-32B-Instruct-AWQ generates physiologically valid `permitted_range` constraints, resolving intervention conflicts (e.g., sodium reduction → compensatory carbohydrate increase)
3. **DiCE** — Diverse Counterfactual Explanations generate 4 personalized health improvement pathways within guardrail boundaries, with automatic **Fallback Logic** for stepwise goal transition

| Risk Class | Definition | Role in Framework |
|---|---|---|
| Class 0 | Normal (no hypertension, no diabetes) | Primary intervention target |
| Class 1 | Hypertension only | Fallback target 1 |
| Class 2 | Diabetes only | Fallback target 2 |
| Class 3 | Comorbid (hypertension + diabetes) | Intervention subject |

---

## Repository Structure

```
.
├── comorbid_risk_cf_pipeline.ipynb   # Full pipeline: preprocessing → training → guardrail → CF generation
├── phd_experiment_results.json       # All experiment outputs (Tables 3, 4, 5 in paper)
├── model_male_v2.pkl                 # Pretrained XGBoost model — male
├── model_female_v2.pkl               # Pretrained XGBoost model — female
├── agent_config_v2.pkl               # vLLM guardrail agent configuration & prompt templates
├── requirements.txt
└── README.md
```

### File Descriptions

**`comorbid_risk_cf_pipeline.ipynb`**  
End-to-end Jupyter notebook covering all experimental steps:
- Section 1: KNHANES data loading & preprocessing
- Section 2: Gender-specific XGBoost model training & evaluation (Accuracy, Recall per class)
- Section 3: vLLM guardrail agent setup & `permitted_range` generation
- Section 4: DiCE counterfactual generation with guardrail injection
- Section 5: Fallback Logic execution (Class 3 → 0 → 1 → 2)
- Section 6: Case study analysis (male & female, Tables 3–5)

**`phd_experiment_results.json`**  
Pre-computed results for all case studies reported in the paper, including:
- Male case: 4 CF pathways to Class 0 (Table 3)
- Female case: 4 CF pathways to Class 1 via Fallback (Table 4)
- Guardrail ablation: w/ vs. w/o guardrail comparison (Table 5)

**`model_male_v2.pkl` / `model_female_v2.pkl`**  
Pretrained XGBoost multi-class models (4-class: 0–3) trained on KNHANES data.  
- Male model accuracy: 0.58 | Class 3 recall: 0.47  
- Female model accuracy: 0.75 | Class 3 recall: 0.59

**`agent_config_v2.pkl`**  
Serialized configuration for the vLLM guardrail agent, including:
- System & user prompt templates
- Conflict prevention rule definitions (sodium–carbohydrate, BMI–waist synchronization, energy floor 70%)
- Gender-specific constraint logic

---

## Requirements

### Experimental Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5-14500 (14 cores / 20 threads, up to 4.32GHz) |
| RAM | 128GB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| OS | Windows 10 / Ubuntu 20.04+ |
| Python | 3.10+ |

### Installation

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
xgboost==2.0.3
scikit-learn==1.4.2
dice-ml==0.11
shap==0.45.0
openai==1.30.1
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
imbalanced-learn==0.12.3
matplotlib==3.8.4
seaborn==0.13.2
tqdm==4.66.4
jupyter==1.0.0
```

> **vLLM** must be installed separately (requires CUDA 11.8+):
> ```bash
> pip install vllm==0.4.2
> ```
> If GPU is unavailable, the pipeline can be run in **guardrail-disabled mode** (pure DiCE, no LLM agent).

---

## Data

This study uses the **Korea National Health and Nutrition Examination Survey (KNHANES)**.

🔗 Official portal: https://knhanes.kdca.go.kr/

Due to data redistribution restrictions, raw KNHANES data is not included in this repository. Please download directly from the official portal and place the file at the path specified in Section 1 of the notebook.

---

## Quickstart

### 1. Start the vLLM server (GPU required)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000
```

### 2. Run the notebook

```bash
jupyter notebook comorbid_risk_cf_pipeline.ipynb
```

Execute cells sequentially. Each section corresponds to a paper section:

| Notebook Section | Corresponds to Paper |
|---|---|
| Section 2: Model training | Section 3.3 |
| Section 3: Guardrail generation | Section 3.4 + Table 2 |
| Section 4: CF generation | Section 3.5 + Tables 3, 4 |
| Section 5: Fallback Logic | Section 3.6 |
| Section 6: Ablation | Section 4.4 + Table 5 |

### 3. Load pre-computed results

To skip computation and inspect reported results directly:

```python
import json
with open("phd_experiment_results.json", "r") as f:
    results = json.load(f)

# Male case (Table 3)
print(results["male_case"]["cf_pathways"])

# Female case with Fallback (Table 4)
print(results["female_case"]["fallback_target"])   # → 1 (Class 1)
print(results["female_case"]["cf_pathways"])

# Guardrail ablation (Table 5)
print(results["ablation"]["without_guardrail"])
print(results["ablation"]["with_guardrail"])
```

---

## Key Clinical Constraints Implemented

| Constraint | Rule | Clinical Rationale |
|---|---|---|
| Sodium–Carbohydrate conflict | When sodium↓, carbohydrate ceiling = current value | Prevents compensatory sugar intake (Feldman & Schmidt, 1999) |
| Physical synchronization | BMI↓ requires waist↓ within plausible range | Physiological co-occurrence |
| Energy floor | Minimum intake ≥ 70% of current | Prevents extreme dietary restriction |
| Fallback sequence | Class 0 → Class 1 → Class 2 | Stepwise realistic goal setting |

---

## Reproducing Paper Tables

```bash
# All results are pre-computed in phd_experiment_results.json
# To re-run from scratch, execute all cells in comorbid_risk_cf_pipeline.ipynb
```

| Result | Location in repo | Paper reference |
|---|---|---|
| Male case 4 CF pathways | `phd_experiment_results.json` → `male_case` | Table 3 |
| Female case Fallback pathways | `phd_experiment_results.json` → `female_case` | Table 4 |
| Guardrail ablation comparison | `phd_experiment_results.json` → `ablation` | Table 5 |
| Model performance (Acc, Recall) | Notebook Section 2 output | Section 3.3 |
| Guardrail JSON examples | Notebook Section 3 output | Table 2 |

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yang2025llm,
  title   = {LLM Agent-Driven Counterfactual Explanations for Comorbid
             Chronic Disease Risk Management: An XAI Framework with Clinical Guardrails},
  author  = {Yang, Myeonggyun and Chun, Hyunwoo},
  journal = {[Journal Name]},
  year    = {2025},
  doi     = {[DOI]}
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

> ⚠️ **Disclaimer**: This repository is intended for academic research purposes only.  
> Clinical application of any outputs requires validation by qualified medical professionals.
