# SkillCraft ML
**University of California, Irvine** · William Chen

Multinomial classification of StarCraft II player skill levels using the SkillCraft1 dataset. Extends Thompson et al. (2013) by addressing class imbalance through three strategies — Baseline, Class Weights, and SMOTE + Tomek Links — across three machine learning models: Multinomial Logistic Regression, Decision Tree, and Random Forest.

## Key Techniques
- 6-class multinomial classification (Leagues 1–6)
- Class imbalance handling: Class Weights, SMOTE + Tomek Links
- Model comparison: Logistic Regression, Decision Tree, Random Forest
- Pairwise binary classification benchmarked against Thompson et al. (2013)
- Feature selection via VIF analysis and backward elimination (AIC)
- Random Forest diagnostics: OOB error convergence, variable importance, MDS proximity plot

## Tools
R • nnet • rpart • randomForest • caret • UBL • ggplot2

## Repository
`skillcraft_analysis.ipynb` — Main analysis notebook

## References

Thompson, J. J., Blair, M. R., Chen, L., & Henrey, A. J. (2013). Video game telemetry
as a critical tool in the study of complex skill learning. *PLOS ONE, 8*(9), e75129.
https://doi.org/10.1371/journal.pone.0075129

Blair, M., Thompson, J., Henrey, A., & Chen, B. (2013). SkillCraft1 Master Table Dataset
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5161N