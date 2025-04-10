1. Introduction
In recent years, ESG (environmental, social, and governance) has garnered much attention from investors and regulators as a key indicator of corporate sustainability. Several studies have been devoted to exploring the predictive effects of financial indicators on ESG scores: for example, Whelan, Atz, and Clark (2021), after reviewing more than 1,000 studies, point out that ESG is generally positively correlated with factors such as operational efficiency, stock returns, and cost of capital. While D'Amato, D' D'Ecclesia, and Levantesi (2021) investigated the relationship between ROE, EBITDA, market capitalisation, and ESG scores using a random forest model. Meanwhile, Friede et al. (2015) found that ‘ESG tends to be positively or insignificantly negatively associated with firms’ financial performance’. Khan et al. (2016) remind us that the industry in which firms operate has a “substantive screening” effect on ESG issues, and that industries differ significantly in terms of their sustainability challenges and regulatory environments, which must be included in the assessment. 
Since many companies do not disclose specific ESG sub-indicators, investors still need to measure the sustainability performance of these companies. This study uses financial metrics (e.g., ROE, EBITDA, market cap) and industry context from S&P 500 companies (via Yahoo Finance), combined with machine learning models, to explore two key questions:
Can financial and industry information effectively predict the ESG scores or ratings of enterprises? 
Do different industries show specific patterns or subgroups in the relationship between ESG and financial data? 



2. Data description
Based on the data provided by Yahoo Finance, the S&P 500 constituent companies were used as the basis for this study, and finally, 437 companies were screened out, with complete data and no obvious deficiencies. There are three main categories of information including:

Category
Indicators
Function
Financial Metrics
ROE 
A measure of a company's profitability, with higher values generally representing stronger shareholder returns
EBITDA 
Reflects the company's core operating performance, excluding the impact of interest, taxes, depreciation, and amortization
Market Cap 
It reflects the scale of the enterprise, which is usually positively related to the status and popularity of the enterprise in the industry
ESG Metrics
esg_score 
The company's comprehensive score in environmental, social, and governance aspects is higher the value, the better the sustainability performance
esg_cat 
According to the total score, enterprises are divided into LAG_PERF (lagging), AVG_PERF (average), LEAD_PERF (leading), and other grades
Industry Classification
Sector 
11 major sectors, e.g,. Energy, Technology, Consumer Defensive, etc.
Industry 
60+ more specific industry categories, e.g., ‘Oil & Gas E&P’, ‘Semiconductors’, ‘Pharmaceuticals ’

3. Exploratory Data Analysis
Preliminary statistics show that the mean total ESG score (esg_score) is around 20.63, with a standard deviation of 6.74. The direct correlation between traditional financial metrics and ESG is relatively limited.

After applying a logarithmic transformation, the correlation between log(EBITDA) and ESG scores increases slightly to 0.17, while log(Market Cap) remains largely uncorrelated. Though statistically weak, this relationship is meaningful in practice, as ESG scores are shaped by numerous qualitative and non-financial factors. A 0.17 correlation suggests that EBITDA explains 17% of the variation in ESG scores, a noteworthy insight given the complexity of ESG evaluation. This highlights the role of profitability in influencing sustainability outcomes. Even if it's not the primary driver, incorporating financial metrics adds value to ESG modeling, supporting better decision-making and reinforcing the link between financial and ESG performance.

When analysed with industry data, we see that industry background has a relatively more significant impact on ESG levels, with sectors like energy, utilities, and basic materials scoring higher, while communication services, real estate, and some consumer sectors score lower. This highlights clear structural differences. We also find that the relationship between ROE and ESG varies significantly across industries (e.g., similar profitability yields different ESG outcomes in energy vs. technology), suggesting that industry acts not just as context but as a moderator in ESG-finance dynamics.

4. Methodology
To solve the problem, 4 models will be fitted for regression analysis of the ESG score, and 4 models will be considered for classification of the ESG category. Naive, Random Forest, XGBoost and Support Vector Machine will be used in both tasks, having the following pipeline:

5. Model Analysis
5.1.1 Random Forest Classification
The best-performing Random Forest classifier used 300 trees, 35 features, and a maximum depth of 9. Random Forest achieved a classification accuracy of 66.7%. The matrix below shows the normalized proportion of correct and incorrect predictions across ESG performance categories.

The model predicted LAG_PERF well (86% accuracy) and did decently on AVG_PERF, but struggled with LEAD_PERF, correctly identifying only 32%. Most top performers were misclassified as average, likely because they share similar financial or industry traits. This suggests more detailed features may be needed to better separate high performers.

The top features were mainly financial: Market Cap, EBITDA, and ROE, with sector variables like Technology and Consumer Defensive also contributing. This shows ESG classification was influenced by both financial size and industry, with financials playing a slightly bigger role.

5.1.2 Random Forest Regression
The Random Forest regressor also used 300 estimators, 35 features, and a tree depth of 9 as identified by grid search.

This histogram shows the prediction errors from the Random Forest regression model. The residuals are centered around zero with a bell-shaped curve, suggesting the model was unbiased and stable. The MAPE of 22.6% makes it one of the better models for predicting ESG scores.

The top features were EBITDA, sector_Real Estate, and sector_Technology, followed by Market Cap and ROE. Industry variables dominated, suggesting that sector identity influences ESG scores more than financial performance, likely due to differing industry norms and expectations.

5.2.1 XGBoost Classification
The XGBoost classifier performed best with 250 estimators, a maximum depth of 6, and a learning rate of 0.09. XGBoost had a lower classification accuracy of 58.3% compared to SVM and Random Forest.

XGBoost predicted LAG_PERF fairly well (77% accuracy), but struggled with LEAD_PERF, only 4% were correctly identified, while most were misclassified as AVG_PERF or LAG_PERF. It also confused 37% of AVG_PERF cases as LAG_PERF. This suggests the model overfits dominant classes and has difficulty separating top ESG performers from the rest.

Top features include Oil & Gas E&P, Entertainment, and Drug Manufacturers. This suggests that the model places heavy emphasis on specific industry types, making it highly sensitive to niche patterns. While this might improve performance on familiar sectors, it could also reduce the model’s ability to generalize well to new or underrepresented industries.

5.2.2  XGBoost regression
The XGBoost regression used 800 estimators, a max depth of 6, and a learning rate of 0.08 for optimal results.

The residuals are centered around zero, indicating that the XGBoost regression model is generally unbiased. The Mean Absolute Percentage Error (MAPE) was approximately 22.5%, comparable to Random Forest.

Similar to its classification, the XGBoost regression relies heavily on industry-related features. Sector_Energy and Sector_Technology top the list, indicating that ESG scores are more strongly influenced by industry affiliation than traditional financial metrics. Financial variables like market capitalization and EBITDA are noticeably absent from the top contributors, suggesting that XGBoost detects patterns within specific sectors that may correlate with ESG outcomes.


5.3.1 SVM Classification
The optimal SVM classifier used a regularization parameter C=1, a radial basis function (RBF) kernel, and gamma set to 'scale'. SVM produced the highest classification accuracy at 67.9%, slightly outperforming Random Forest.

The matrix shows strong classification accuracy for LAG_PERF and AVG_PERF, similar to other models. However, where SVM stands out is its improved handling of LEAD_PERF, correctly predicting 40% of top-performing instances. This balanced performance across categories suggests that SVM effectively captures non-linear patterns in the data and may generalize better to edge cases like LEAD_PERF.

The SVM model relied most on industry-specific features like Oil & Gas E&P, Drug Manufacturers, and Medical Devices, showing its sensitivity to detailed sector differences. This likely helped it perform better on LEAD_PERF, as niche industry traits may distinguish top ESG performers more than financials do.

5.3.2 SVM Regression
The best SVM regressor used C=10, a polynomial kernel, and gamma set to 'scale'. SVM also achieved the lowest MAPE of 22.1%, making it the best model for regression as well.

The residuals are centered around zero, indicating that the model is generally unbiased.

The SVM regressor heavily prioritizes industry-specific features, with Oil & Gas E&P, Aerospace & Defense, and Conglomerates being top contributors. This reflects SVM’s tendency to capitalize on categorical sector dummies, using them to learn complex ESG performance signals that aren't necessarily captured by financial metrics.

5.4 Naive Baseline
As a benchmark:
Classification Accuracy: 47.4%
Regression MAPE: 31.4%
This confirms that machine learning significantly improves ESG prediction over basic heuristics.


5.5 Model Comparison Summary 
Model
Classification Accuracy
Regression MAPE
Top Features
Strengths
SVM
67.9%
22.1%
Industry, Sector
Best overall classifier & Regressor
Random Forest
66.7%
22.6%
EBITDA, Sector, Market Cap
Balanced, Interpretable
XGBoost
58.3%
22.5%
Sector-specific
Good sector Insights, Less Generalizable
Naive
47.4%
31.4%
—
Baseline only



6. Conclusions and prospects
After a comprehensive analysis of ESG data and financial indicators of the companies, we mainly draw the following conclusions:
a. Machine learning models are better predictors than random. All 6 models perform better than Naive models, with the regression models having lower MAPE and the classification models, higher accuracy. This difference is more pronounced in classification, where Naive can predict accurately only 47% of the time; meanwhile, on regression, the errors are  32% of the magnitude of the ESG score, which is still low.
b. Support Vector Machine is the most accurate model in regression and classification. With a 68% accuracy and a 22.13% MAPE, SVM performs better than Random Forest and XGBoost. However, the differences between them are low, which means that it can be influenced by the randomness of the models.
c. Classification models are more accurate, but regression models are less biased. When compared to Naive models, there is more improvement in classification models; however, they struggle to predict lead performers. Regression models do not improve the accuracy of the Naive model in a big way, but the residuals are centered at 0, making them unbiased.
d. In terms of predictive modeling, we found that "industry factors" are the main variables that determine ESG performance. Taking the SVM models as an example, the accuracy of the model on the test set reaches 68%, and the MAPE 22.13%, indicating that the model itself has certain predictive ability, but further feature importance analysis shows that the improvement of this accuracy mainly comes from the identification of the industry to which the enterprise belongs.
