# SkillCraft ML
Multinomial classification of StarCraft II player skill levels (6 leagues, Bronze through Masters) using the SkillCraft1 dataset. Extends Thompson et al. (2013) by tackling the full 6-class problem, achieving 40.7% accuracy with RF + Class Weights and outperforming their Conditional Inference Forest in 3 out of 4 pairwise comparisons.

## Motivation
UC Irvine is surrounded by game companies, and player skill classification is a genuinely useful problem for the industry: understanding what separates skill levels helps inform how ranking systems are calibrated and whether in-game mechanics need adjustment. The SkillCraft1 dataset and Thompson et al. (2013) made this a clean starting point, with behavioral telemetry from 3,395 StarCraft II players already collected and a published baseline to compare against.

## Design Decisions

**Why three imbalance strategies?**

In a 6-class problem, Bronze players make up only 5% of the dataset. The "right" strategy for handling this imbalance isn't obvious upfront: Baseline ignores it entirely, Class Weights adjusts the loss function without touching the data, and SMOTE + Tomek Links generates synthetic minority samples while cleaning decision boundaries.

**Why these three models?**

Logistic Regression extends naturally from binary to multinomial classification, making it a clean baseline for this 6-class problem. Decision Tree captures nonlinearity but is prone to overfitting. Random Forest aggregates many trees for stability. Comparing all three reveals how much complexity is actually needed.

**Why replicate Thompson et al.'s pairwise setup?**

To test whether handling class imbalance actually improves performance on the same benchmark. Replicating their pairwise setup on equal terms gives a direct answer.

## Key Results
RF + Class Weights achieves the best overall balance. Given severe class imbalance and substantial behavioral overlap between adjacent leagues, 40.7% accuracy with meaningful minority recall is a reasonable result for this problem.

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

ActionLatency and APM are the two most important predictors, consistent with Thompson et al.'s finding that action speed dominates skill prediction. Demographic features (Age, HoursPerWeek) rank low, suggesting in-game behavior matters more than time invested.

## Reflections & Next Steps

Class imbalance strategies involve real tradeoffs. SMOTE + Tomek maximized minority recall but collapsed mid-tier predictions in Decision Trees. Class Weights was more reliable because it adjusts incentives without fabricating data.

The 40% accuracy ceiling reflects a harder problem: adjacent leagues share very similar behavioral profiles, and no amount of resampling fixes a signal problem. The MDS proximity plot confirms this. League 1 separates cleanly, but Leagues 2 through 4 overlap substantially.

Next steps:
- **Gradient boosting**: XGBoost handles class imbalance natively and often outperforms Random Forest on tabular data.
- **Longitudinal analysis**: tracking the same players over time would reveal how behavioral patterns evolve with skill development, rather than relying on cross-sectional snapshots.

## Repository

- `report/skillcraft_report.pdf`: Final report
- `code/skillcraft_analysis.ipynb`: Main analysis notebook
- `code/skillcraft_analysis.r`: Executable R script
- `figures/`: All plots and visualizations

## Tools

R · nnet · rpart · randomForest · caret · smotefamily · themis · ggplot2

## References

Thompson, J. J., Blair, M. R., Chen, L., & Henrey, A. J. (2013). Video game telemetry as a critical tool in the study of complex skill learning. *PLOS ONE, 8*(9), e75129. https://doi.org/10.1371/journal.pone.0075129

Blair, M., Thompson, J., Henrey, A., & Chen, B. (2013). SkillCraft1 Master Table Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5161N