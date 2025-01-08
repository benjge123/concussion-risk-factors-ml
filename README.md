<<<<<<< HEAD
# Concussion Risk ML Project

Go to 'Code" to view with formatting.

1/6/2024
1. Removed PCS1-22 and MFQ1-33 since the ML model was cheating/overfitting since it could add up scores for both evaluation metrics and that would get you the PCS and MFQ severity score
2. Reran the regressions to get the following: PCS Model - MAE: 10.88, RMSE: 17.16, R²: 0.46
					       MFQ Model - MAE: 4.14, RMSE: 5.54, R²: 0.81
3. Evaluated feature importance of my X variables in regression to improve model with OLS analysis. Used 0.05 as p-value. r^2 went up to 0.621 after removing variables.
4. Compared training vs test r^2 for PCS model. 0.66 vs 0.50. Some overfitting still exists.
5. Checked for Multi-collinearity with VIF analysis. Age had a correlation greater than 14. I checked what other variables it's highly correlated with and Sport had a correlation of 0.4. Not sure why the multicollinearity was so high but I removed it since including it only slightly improved the model performance (.02 increase in adj r^2).
6. Since I identified slight overfitting previously, I tried Ridge and Lasso regression to see if it could help. Neither did. Lasso regression performed the same as linear so I'll keep using the linear regression.
7. Since I think I have the best PCS model, I moved onto ranking how important each variable is.
8. Going back to the MFQ model, PCS and MFQ scores had a 0.78 correlation. Since the PCS already includes some questions regarding emotional well-being, I decided to drop the MFQ model entirely and focus on the PCS Model. The goal I had when I wrote the MIT Essay question was to rank risk factors for brain injuries and the PCS model is the most relevant for that.
9. Partially reduced overfitting with training r^2 = 0.62 and testing r^2 = 0.52.
10. Feature importance ranked for PCS Severity Score
	MFQ Cutoff(Emotionally Distressed? y/n)	33.74
	Learning Disability			15.21
	Aggregate Medical History		13.89
	Anxiety Diagnosis			12.90
	Depression Diagnosis			12.32
	Anxiety Symptoms			9.95
	Prior Depressive Episodes (Y/N)		6.19
	Sex					4.76
	# of Prior Depressive Episodes		0.27
11. Created a residual plot and saw that my regression model struggled to predict severe cases. No surprise since there are only three cases with a PCS score higher than 80. The max score is 132.


7. Figure out why my README file isn't updating with Git. I had to manually paste this.


To Consider Later
1. Group MFQ (Core depressive, core Anxiety) and PCS Symptoms (Cognitive, physical, emotional) into categories. This could help uncover patterns without data leakage from 1st regression initiation with huge overfitting. 
	-Can help model understand what category of symptoms impact severity the most.
2. Answer "Does history of concussion, anxiety, depression, lead to worse MFQ/PCS?
3. "So given your answers to your past medical history, PCS, and MFQ scores, here's a loose projection of what your near future could look like" - if this model were to be turned into a tool to be used in practice, this is a situation where it could help a patient have a more detailed understanding of their next few weeks. It beats "try to rest and cross your fingers you get better" 
=======
# Concussion Risk ML Project

Go to 'Code" to view with formatting.

1/6/2024
1. Removed PCS1-22 and MFQ1-33 since the ML model was cheating/overfitting since it could add up scores for both evaluation metrics and that would get you the PCS and MFQ severity score
2. Reran the regressions to get the following: PCS Model - MAE: 10.88, RMSE: 17.16, R²: 0.46
					       MFQ Model - MAE: 4.14, RMSE: 5.54, R²: 0.81
3. Evaluated feature importance of my X variables in regression to improve model with OLS analysis. Used 0.05 as p-value. r^2 went up to 0.621 after removing variables.
4. Compared training vs test r^2 for PCS model. 0.66 vs 0.50. Some overfitting still exists.
5. Checked for Multi-collinearity with VIF analysis. Age had a correlation greater than 14. I checked what other variables it's highly correlated with and Sport had a correlation of 0.4. Not sure why the multicollinearity was so high but I removed it since including it only slightly improved the model performance (.02 increase in adj r^2).
6. Since I identified slight overfitting previously, I tried Ridge and Lasso regression to see if it could help. Neither did. Lasso regression performed the same as linear so I'll keep using the linear regression.
7. Since I think I have the best PCS model, I moved onto ranking how important each variable is.
8. Going back to the MFQ model, PCS and MFQ scores had a 0.78 correlation. Since the PCS already includes some questions regarding emotional well-being, I decided to drop the MFQ model entirely and focus on the PCS Model. The goal I had when I wrote the MIT Essay question was to rank risk factors for brain injuries and the PCS model is the most relevant for that.
9. Partially reduced overfitting with training r^2 = 0.62 and testing r^2 = 0.52.
10. Feature importance ranked for PCS Severity Score
	MFQ Cutoff(Emotionally Distressed? y/n)	33.74
	Learning Disability			15.21
	Aggregate Medical History		13.89
	Anxiety Diagnosis			12.90
	Depression Diagnosis			12.32
	Anxiety Symptoms			9.95
	Prior Depressive Episodes (Y/N)		6.19
	Sex					4.76
	# of Prior Depressive Episodes		0.27
11. Created a residual plot and saw that my regression model struggled to predict severe cases. No surprise since there are only three cases with a PCS score higher than 80. The max score is 132.



To Consider Later
1. Ignore - Group MFQ (Core depressive, core Anxiety) and PCS Symptoms (Cognitive, physical, emotional) into categories. This could help uncover patterns without data leakage from 1st regression initiation with huge overfitting. 
	-Can help model understand what category of symptoms impact severity the most.
2. Answer "Does history of concussion, anxiety, depression, lead to worse PCS?
3. "So given your answers to your past medical history, PCS, and MFQ scores, here's a loose projection of what your near future could look like" - if this model were to be turned into a tool to be used in practice, this is a situation where it could help a patient have a more detailed understanding of their next few weeks. It beats "try to rest and cross your fingers you get better" 
4. Should I remove the cases where PCS = 0? There's 43/155 who do not have and Post Concussion Sypmtoms, so I wonder if it's making my model worse by have the largest population of the data be people with 0 symptoms. I am also unsure if my SMOTE over sampling is working the way it should....
>>>>>>> eeea1f2 (Save local changes before rebase)
