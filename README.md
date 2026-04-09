# SkillCraft ML

Multinomial classification of StarCraft II player skill levels (6 leagues, Bronze through Masters) using the SkillCraft1 dataset. Extends Thompson et al. (2013) by tackling the full 6-class problem and comparing three class imbalance strategies across three models.

## Motivation

Thompson et al. (2013) studied how variable importance shifts across StarCraft II skill levels using pairwise binary classifiers. Their approach was deliberately scoped: compare leagues two levels apart, one pair at a time. This sidesteps the harder question of whether you can classify all six leagues simultaneously, especially when class distribution is severely imbalanced (Bronze is only 5% of the dataset).

This project takes on that harder question. The goal was to see how far standard ML models can go on a 6-class problem with real class imbalance, and whether imbalance-aware training strategies actually help or just trade one problem for another.

## Design Decisions

**Why three imbalance strategies?**

Because the "right" strategy isn't obvious upfront. Baseline ignores imbalance entirely. Class Weights adjusts the loss function without touching the data. SMOTE + Tomek Links generates synthetic minority samples and cleans decision boundaries. Comparing all three on the same models reveals the actual tradeoff: SMOTE + Tomek consistently maximized minority recall (up to 0.91) but at significant accuracy cost, while Class Weights offered the most practical balance.

**Why these three models?**

Logistic Regression gives interpretable coefficients and a linear baseline. Decision Tree captures nonlinearity but is prone to overfitting. Random Forest aggregates many trees for stability. The comparison is deliberate: if Random Forest only marginally beats logistic regression, the nonlinear patterns aren't strong enough to justify the complexity.

**Why replicate Thompson et al.'s pairwise setup?**

To benchmark against the original study on equal terms. Our RF + Class Weights outperformed their Conditional Inference Forest in 3 out of 4 league pairs, with the only loss on Gold vs. Diamond, the hardest boundary where mid-tier players behave most similarly.

## Key Results

| Algorithm | Strategy | Accuracy | Minority Recall |
|-----------|----------|----------|-----------------|
| Logistic Regression | Baseline | 0.415 | 0.273 |
| Logistic Regression | Class Weights | 0.381 | 0.515 |
| Decision Tree | SMOTE + Tomek | 0.284 | 0.909 |
| **Random Forest** | **Class Weights** | **0.407** | **0.333** |
| Random Forest | SMOTE + Tomek | 0.393 | 0.667 |

RF + Class Weights achieves the best overall balance. In a 6-class problem where random chance is 16.7%, an accuracy of 40.7% with meaningful minority recall represents a reasonable result given the inherent difficulty of separating adjacent leagues.

ActionLatency and APM are the two most important predictors, consistent with Thompson et al.'s finding that action speed dominates skill prediction. Demographic features (Age, HoursPerWeek) rank low, suggesting in-game behavior matters more than time invested.

## Reflections & Next Steps

The clearest takeaway: class imbalance strategies involve real tradeoffs, not free improvements. SMOTE + Tomek pushed minority recall to 0.91 but collapsed mid-tier league predictions entirely in Decision Trees. Class Weights was the most reliable strategy because it adjusts incentives without fabricating data.

The 40% accuracy ceiling also reveals a fundamental challenge. Adjacent leagues share very similar behavioral profiles. The MDS proximity plot confirms this: League 1 separates cleanly, but Leagues 2 through 4 overlap substantially in the feature space. No amount of resampling fixes a signal problem.

Next steps:
- **Ordinal regression**: the current models treat leagues as unordered categories, but league rank has inherent ordering. Ordinal logistic regression could exploit this structure.
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