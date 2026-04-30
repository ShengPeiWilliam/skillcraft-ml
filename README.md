# Multi-Class Skill Classification in StarCraft II

[![Full Report](https://img.shields.io/badge/📄_Read_Full_Report-PDF-blue?style=for-the-badge)](report/skillcraft_report.pdf)

Multinomial classification of StarCraft II player skill levels (6 leagues, 
Bronze through Masters) on 3,340 players from the SkillCraft1 dataset.

This project extends Thompson et al. (2013), who used pairwise binary 
classification with Conditional Inference Forest, by tackling the harder 
full 6-class problem and benchmarking three class-imbalance strategies. 
Random Forest with class weighting achieves 40.7% accuracy and outperforms 
Thompson et al.'s baseline in 3 of 4 pairwise league comparisons (e.g., 
Bronze–Gold 88.2% vs 82.3%), demonstrating that cost-sensitive learning 
matches resampling-based methods without synthetic data risks.

## Motivation

UC Irvine is surrounded by game companies, and player skill classification 
is a genuinely useful problem for the industry: understanding what separates 
skill levels helps inform how ranking systems are calibrated and whether 
in-game mechanics need adjustment. The SkillCraft1 dataset and Thompson 
et al. (2013) made this a clean starting point, with behavioral telemetry 
from 3,395 StarCraft II players already collected and a published baseline 
to compare against.

## Design Decisions

**Why three imbalance strategies?**

In a 6-class problem, Bronze players make up only 5% of the dataset. The 
"right" strategy for handling this imbalance isn't obvious upfront: Baseline 
ignores it entirely, Class Weights adjusts the loss function without touching 
the data, and SMOTE + Tomek Links generates synthetic minority samples while 
cleaning decision boundaries.

**Why these three models?**

Logistic Regression extends naturally from binary to multinomial 
classification, making it a clean baseline for this 6-class problem. 
Decision Tree captures nonlinearity but is prone to overfitting. Random 
Forest aggregates many trees for stability. Comparing all three reveals 
how much complexity is actually needed.

**Why replicate Thompson et al.'s pairwise setup?**

Their original work used pairwise binary classifiers with Conditional 
Inference Forest, achieving 78–82% accuracy across league pairs. To test 
whether modern class-imbalance strategies actually improve performance on 
the same benchmark, replicating their setup on equal terms gives a direct 
answer rather than relying on overall multinomial accuracy as a proxy.

**Why select Random Forest with Class Weights as the final model?**

Logistic Regression Baseline had the highest raw accuracy (41.5%) but only 
27% minority recall — it was effectively ignoring the minority class. SMOTE 
+ Tomek pushed minority recall higher but collapsed mid-tier predictions 
in Decision Trees and dropped overall accuracy substantially. Random Forest 
with Class Weights struck the best balance: 40.7% accuracy with meaningful 
minority recall, without the synthetic-data risks of resampling.

## Key Results

**Headline finding**: RF + Class Weights outperformed Thompson et al.'s Conditional Inference Forest in 3 of 4 pairwise league comparisons (Bronze–Gold 88.2% vs 82.3%, Silver–Platinum 81.9% vs 78.3%, Platinum–Masters 84.7% vs 80.7%), demonstrating that cost-sensitive learning is a practical alternative to resampling.

**6-class multinomial benchmark**

| Algorithm | Strategy | Accuracy | Minority Recall |
|-----------|----------|----------|-----------------|
| Logistic Regression | Baseline | 0.415 | 0.273 |
| Logistic Regression | Class Weights | 0.381 | 0.515 |
| Logistic Regression | SMOTE + Tomek | 0.366 | 0.909 |
| Decision Tree | Baseline | 0.377 | 0.455 |
| Decision Tree | Class Weights | 0.287 | 0.394 |
| Decision Tree | SMOTE + Tomek | 0.284 | 0.909 |
| Random Forest | Baseline | 0.402 | 0.333 |
| **Random Forest** | **Class Weights** | **0.407** | **0.333** |
| Random Forest | SMOTE + Tomek | 0.393 | 0.667 |

**Pairwise comparison vs Thompson et al. (2013)**

| League Pair | Thompson et al. (CForest) | Ours (RF + CW) |
|-------------|---------------------------|----------------|
| Bronze–Gold | 82.3% | **88.2%** |
| Silver–Platinum | 78.3% | **81.9%** |
| Gold–Diamond | **79.0%** | 75.0% |
| Platinum–Masters | 80.7% | **84.7%** |

ActionLatency and APM are the two most important predictors, consistent with 
Thompson et al.'s finding that action speed dominates skill prediction. 
Demographic features (Age, HoursPerWeek) rank low, suggesting in-game behavior 
matters more than time invested.

## Reflections & Next Steps

Class imbalance strategies involve real tradeoffs. SMOTE + Tomek maximized 
minority recall but collapsed mid-tier predictions in Decision Trees. Class 
Weights was more reliable because it adjusts incentives without fabricating 
data.

The 40% accuracy ceiling reflects a harder problem: adjacent leagues share 
very similar behavioral profiles, and no amount of resampling fixes a signal 
problem. The MDS proximity plot confirms this. League 1 separates cleanly, 
but Leagues 2 through 4 overlap substantially. The Gold–Diamond pairwise 
result (where our model underperformed Thompson et al.) reflects the same 
limitation: mid-tier league boundaries are inherently the hardest to draw.

Next steps:
- **Gradient boosting**: XGBoost handles class imbalance natively and often outperforms Random Forest on tabular data.
- **Ordinal regression**: league rank has a natural ordering (Bronze < Silver < ... < Masters) that multinomial classification ignores. Ordinal models would respect this structure and may improve mid-tier separation.
- **Longitudinal analysis**: tracking the same players over time would reveal how behavioral patterns evolve with skill development, rather than relying on cross-sectional snapshots.

## Repository

```
report/
└── skillcraft_report.pdf       # Full analysis writeup
code/
├── skillcraft_analysis.ipynb   # Main analysis (R notebook)
└── skillcraft_analysis.R       # Clean R script
figures/                        # All plots and visualizations
```

## Tools

**Statistical methods**: Multinomial Logistic Regression, Decision Tree, Random Forest, SMOTE + Tomek Links, Class Weighting, VIF diagnostics  
**Language**: R  
**Libraries**: nnet, rpart, randomForest, caret, smotefamily, themis, ggplot2

## References

Thompson, J. J., Blair, M. R., Chen, L., & Henrey, A. J. (2013). [Video game telemetry as a critical tool in the study of complex skill learning](https://doi.org/10.1371/journal.pone.0075129). *PLOS ONE, 8*(9), e75129.

Blair, M., Thompson, J., Henrey, A., & Chen, B. (2013). [SkillCraft1 Master Table Dataset](https://archive.ics.uci.edu/dataset/272/skillcraft1+master+table+dataset) [Dataset]. UCI Machine Learning Repository.
