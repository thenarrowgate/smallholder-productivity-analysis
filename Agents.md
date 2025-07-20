Project goal:

Pre-process Nepal and Senegal data, then separately for each do the following: use EFA to distill observed variables into a smaller set of interpretable

underlying factors, and then regress the productivity index on the latent factors scores + other dummies if necessary through an CFA/SEM model to test

the fit of the model. This will then finally provide us with the latent factors underlying productivity in Nepal and in Senegal separately.



Project Structure:

The data directory contains the datasets being used in this project, the subdirectory within the data directory that is called

raw\_updated\_names contains the dataset used by the EDA.ipynb file. The EDA.ipynb file is a python notebook that does some data analysis

and also important preprocessing steps such as dealing with missing values all columns, handling outliers, etc. It finally saves the preprocessed data in the nepal\_dataframe\_FA.xlsx and senegal\_dataframe\_FA.xlsx files which are used by MainMethod.R. MainMethod.R as of now splits the different features

in the data by their variable types (continuous, ordinal, binary, nominal), creates a mixed correlation matrix for EFA, uses parallel analysis and MAP

to get upper and lower bound respectively on k (number of factors to extract), and then performs 1000 EFA factor extractions and rotations on bootstrapped

samples with replacement, and the final loadings / unique variances are the median of all of these bootstraps. It then prunes items whose loadings are unstable and whose communality is low. Then as of now it's final step is performing a 1000 bootstrap test of congruence by recalculating Tucker's phi and Hancock's H between the EFA on this sample and the final loadings, and then prints the mean of these two indices for each of the k factors.



Current sub-goal in the project pipe-line:

As of now, the current goal is to get a stable factor solution (Tucker's phi >= 0.90 and Hancock's H >= 0.80 for all factors).

