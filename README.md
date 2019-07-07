# Titanic Model using Python - An exploration
This is my project to learn how to use Python for data science, with a focus on sklearn, Pandas, and numpy.

## Preprocessing

### Down the Pipeline Rabbit Hole
sklearn.pipeline.Pipeline is a powerful tool for automating the data science workflow. It enables cobbling together feature extraction, feature selection, and feature manipulations with models and then perform hyper-parameter tuning with cross validation in a few function calls.

The struggle is that each step happens sequentially with each feature step returning only those features derived in that step. So say you'd like to apply PCA to the numeric features, one hot encoding to the categorical, and then return the 5 highest PCA components, all of the encoded features, and all of the original numeric features. Well, a simple Pipeline won't do that. You might only get the encoded features at the end.

Additionally, say you want to incorporate some ad hoc feature engineering. For example with the Titanic data, I decided to split the cabin number into two features, the letter/character string, and the actual number. They'll probably turn out useless, but you never know. This isn't a standard feature extraction step, and I'll detail it farther down. However, the string that gets extracted needs to be one hot encoded, but this isn't defined yet when the data is initially loaded, so the column doesn't exist. We'd want the Pipeline to be able to select all charater features and apply encoding to that as-of-yet unknown set of features. This required implementing a little function to select out columns of specific type, pass them as **kwargs to Pandas select_dtypes helper function, so that the transformer gets applied to columns of the appropriate types.

This brought up the investigation into additional Pipeline-esque functions in sklearn, namely ColumnTranformer and FeatureUnion. The former allows the application of a transformation function to a list of columns, while the latter simultaneously applies tranformers/Pipelines to a dataset. Because ColumnTransformer requires columns to be passed to it on an a priori basis, I decided it wasn't appropriate to use. FeatureUnion however was, as I used my MySelector to choose the numeric columns, then union those to a pipeline that one hot encodes all of the object columns, which selects those columns after passing through the preliminary feature engineering step.

### Final Pre-processing pipeline
My completed Pipeline performs a few feature extractions--appends on the cabin letter and number from that feature, drops ticket number, passenget ID, name, and cabin. I then use a simpleImputer to impute numerics with the mean and categorical with 'missing'. I then one-hot-encode the categorical. I might experiment with label-encoding rather than OHE at a later time.

#### PandasFeatureUnion
Another challenge with using Pipelines, FeatureUnions, and ColumnTransformers is that FeatureUnion returns a numpy ndarray. So the Pipeline I have: 'perform data set pre-processing -> impute both numeric & categorical missings -> one-hot-encode categorical' fails because imputing numeric/categorical requires using FeatureUnion and the one-hot-encoding requires extracting the categorical and then FeatureUnion back with the numerics, but the selector requires a pandas.DataFrame rather than and ndarray. I found this nice class, [PandasFeatureUnion](https://github.com/marrrcin/pandas-feature-union) which would resolve my issue. However, I also realized that I can simply cast the ndarry from the FeatureUnion as a DataFrame, so I don't need make use of a separate class. I'm making use of the '''pd.DataFrame.from_records()''' method That said, I'm still going to research how to create Python packages based off of this repository.

## Modelling
I ran into some issues with my pre-processing pipeline, but I think it makes sense to separate the two. The utility of pipelines is two-fold: 1) apply the same set of transformations to train and test data sets and 2) use them in cross validation for transformations that are not defined. With this in mind, it makes sense to have separate pipelines that take the output from the pre-processing step and then apply various transformations for modelling that can be cross-validated.

### Logistic Regression

## TODO
* Create Python package for [PandasFeatureUnion](https://github.com/marrrcin/pandas-feature-union) 
