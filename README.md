# LMM_JM_COVID19
Linear Mixed effect Model using Joint Modeling framework for COVID19 severity 

(Pulmonary, Renal and Immune dysfunction)

---

### The data set is available at CODA(Clinical and Omics Data Archive)

provided by the NIH (Korean National Institutes of Health):

https://coda.nih.go.kr/

---

### Baseline
For the reproducibility, we commit the codes

R installation is required.

```
# from a terminal where the environment exists
conda activate lmm_jm
conda env export --from-history --name lmm_jm > environment.yml
```
Project directory structure
```
/home/project
└─ code/
  ├─ 01_lasso.ipynb
  ├─ 02_lmm.ipynb
  ├─ 03_bootstrap_reproduction.ipynb
  └─ model.py
└─ result/
  ├─ lmin_total.csv
  └─
└─ simulation_mimic_data/
  └─ 106_total


```

---
### Abstract
Although various severity prediction models for COVID-19 have been developed to guide clinical decision-making, substantial inter-individual variation in immune responses results in inconsistent disease trajectories. This variability contributes to clinical heterogeneity and poses challenges for standardized treatment strategies. 

To reflect the complex and systemic nature of COVID-19 severity, we selected three physiologically distinct yet interrelated biomarkers—oxygen saturation, C-reactive protein (CRP), and blood urea levels—as response variables representing pulmonary, inflammatory, and renal function, respectively.

We examined the associations between adaptive immune profiles and three longitudinal severity response variables measured at three time points by employing linear mixed-effects model (LMM) using Joint Modeling framework (JM). This approach handles longitudinal outcomes by incorporating fixed effects, which capture overall population trends, and random effects, which capture variability unique to each individual .

---


