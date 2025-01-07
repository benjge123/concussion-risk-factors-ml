# Concussion Risk ML Project

1/6/2024
1. Removed PCS1-22 and MFQ1-33 since the ML model was cheating/overfitting since it could add up scores for both evaluation metrics and that would get you the PCS and MFQ severity score
2. Reran the regressions to get the following: PCS Model - MAE: 10.88, RMSE: 17.16, R²: 0.46
					       MFQ Model - MAE: 4.14, RMSE: 5.54, R²: 0.81
3. Evaluated feature importance of my X variables in regression to improve model. Used 0.05 as p-value. r^2 went up to 0.621 after removing variables.
4. Compared training vs test r^2 for PCS model. 0.66 vs 0.50. Some overfitting still exists.
5. Checked for Multi-collinearity with VIF analysis. Age had a correlation greater than 10. Not sure why but I removed it.
6. Need to adjust variables to MFQ model? Maybe not since it's doing well. Need to see train vs test r^2 first.
7. Figure out why my README file isn't updating with Git. I had to manually paste this.


To Consider Later
1. Group MFQ (Core depressive, core Anxiety) and PCS Symptoms (Cognitive, physical, emotional) into categories. This could help uncover patterns without data leakage from 1st regression initiation with huge overfitting. 
	-Can help model understand what category of symptoms impact severity the most.
2. Answer "Does history of concussion, anxiety, depression, lead to worse MFQ/PCS?
3. "So given your answers to your past medical history, PCS, and MFQ scores, here's a loose projection of what your near future could look like" - if this model were to be turned into a tool to be used in practice, this is a situation where it could help a patient have a more detailed understanding of their next few weeks. It beats "try to rest and cross your fingers you get better" 
