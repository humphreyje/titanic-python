#Titanic Model using Python - An exploration
This is my project to learn how to use Python for data science, with a focus on sklearn, Pandas, and numpy.

##Preprocessing

###Down the Pipeline Rabbit Hole
sklearn.pipeline.Pipeline is a powerful tool for automating the data science workflow. It enables cobbling together feature extraction, feature selection, and feature manipulations with models and then perform hyper-parameter tuning with cross validation in a few function calls.

The struggle is that each step happens sequentially with each feature step returning only those features derived in that step. So say you'd like to apply PCA to the numeric features, one hot encoding to the categorical, and then return the 5 highest PCA components, all of the encoded features, and all of the original numeric features. Well, a simple Pipeline won't do that. You might only get the encoded features at the end.

Additionally, say you want to incorporate some ad hoc feature engineering. For example with the Titanic data, I decided to split the cabin number into two features, the letter/character string, and the actual number. They'll probably turn out useless, but you never know. This isn't a standard feature extraction step, and I'll detail it farther down. However, the string that gets extracted needs to be one hot encoded, but this isn't defined yet when the data is initially loaded, so the column doesn't exist. We'd want the Pipeline to be able to select all charater features and apply encoding to that as-of-yet unknown set of features. This required implementing a little function to select out columns of specific type, pass them as **kwargs to Pandas select_dtypes helper function, so that the transformer gets applied to columns of the appropriate types.
