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
11. Created a residual plot and saw that my regression model struggled to predict severe cases. No surprise since there are only three cases with a PCS score higher than 80. The max score is 132. Tried SMOTE resampling to help with oversampling. I then realized there were 43 cases where the PCS score was 0, so I tried undersampling too. I got a little stuck here because I don't know what a properly over and undersampled case is supposed to look like for this data. I think if I solve the resampling issue, I can fix the heteroskedascity issue where most of the cases are aggregated in the low PCS score area
12. Started on my decision tree model MAE: 9.91, RMSE: 19.34, R²: 0.31. So far it's not better than the linear regression.
13. Tried cross validation on the DT to see what would happen. R^2 dropped to 0.14, once again reinforcing that my model is overfitting.
14. Moved onto a random forest. Given the overfitting on a small dataset, I doubt random forests would perform better but I have nothing better to do. 
15. No surprise, it has the same r^2. Standard deviation for the RF model went from 0.62->0.08 when compared with the DT model so that's an improvement.
16. Started on polynomial regression to help see the bias-variance tradeoff and find what degree polynomial results in the best polynomial regression.
17. Got stuck on if my polynomial regression model is correct.
18. Finished my LOOCV function, i think. Have yet to test it.
19. Tested LOOCV function of degree 4 polynomial regress. Got a RMSE of 7.9605 compared to train-test-split RMSE of 10.85. Improvement, yes?
20. LOOCV on my RF model had better results too, with a RMSE of 19.78 and r^2 of 0.28. r^2 is a double what it was previously. I didn't record what RMSE of non-LOOCV model was.
21. LOOCV on DT model also improved it significantly. New RSME: 17.88 and r^2: 0.41.
22. Couldn't figure out how to correctly plot a learning curve for LOOCV poly regression
23. Built a classification DT model. 5 classes with none, mild, moderate, severe, and very severe classes. Uses grid search to help find optimal weights for the classes since there's many "none" data points and few severs/very severe data points. Model still meh, 0.55 accuracy after messing with class weights
24. Built a RF model and got a 0.5 accuracy. DT and RF still struggle to predict on the severe and very severe classes, with both being unable to predict them even after heavily weighting their classes



22. How do i go about ordering my code? I now have different models in different files so what's best for combining everything?
19. Got a new laptop, do I need to add this laptop to GitHub or something?

12. I just realized that you recommend I create a separate file with plots from my models, right?



To Consider Later
1. Ignore - Group MFQ (Core depressive, core Anxiety) and PCS Symptoms (Cognitive, physical, emotional) into categories. This could help uncover patterns without data leakage from 1st regression initiation with huge overfitting. 
	-Can help model understand what category of symptoms impact severity the most.
2. Answer "Does history of concussion, anxiety, depression, lead to worse PCS?
3. "So given your answers to your past medical history, PCS, and MFQ scores, here's a loose projection of what your near future could look like" - if this model were to be turned into a tool to be used in practice, this is a situation where it could help a patient have a more detailed understanding of their next few weeks. It beats "try to rest and cross your fingers you get better" 
4. Should I remove the cases where PCS = 0? There's 43/155 who do not have and Post Concussion Sypmtoms, so I wonder if it's making my model worse by have the largest population of the data be people with 0 symptoms. I am also unsure if my SMOTE over sampling is working the way it should....
5. Is it bad that MFQ cut off is included since MFQ scores has a 0.78 with PCS scores? If removed, r^2 drops to ~0.11, far worse than ~0.5
6. If I still want to do resampling, Leave one out cross validation can be done first
7. Jen Weekend Email Plan
	-Take a look at data from different angles to better understand questions I can answer w data
	-Think about normalizing or standardizing features
	-Try feature engineering and retry the polynomial regression w feature engineering
		-Figure out what linear algerbra she's talking about means
	-Give classification a shot. Read the textbook about classification first?
	-Figure out what the criteria of "What a working model" is
	-Can switch to answering her short answer questions if I get stuck or need a break

