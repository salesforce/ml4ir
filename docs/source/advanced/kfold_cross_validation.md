## Running Kfold Cross Validation

ml4ir allows to run in a K-fold Cross validation mode. This mode reads the data the same way
as the normal "non K-fold" mode and merges the data sets training, validation and test (if specified) together.
Then according to the specified number of folds the merged data set is split among the training, validation and test sets.
  

You can control the K-fold mode by specifying three additional command line arguments.

1) kfold

The number of folds for K-fold Cross Validation. Must be > 2 if testset is included in folds and > 1 otherwise.

2) include_testset_in_kfold

Whether to merge the testset with the training and validation sets and perform kfold on the merged dataset.

3) monitor_metric

Metric to use for post Kfold CV analysis.



Example
```
--kfold 5
--kfold_analysis_metrics MRR
--include_testset_in_kfold False
```

This would split the dataset into 5 folds: f1, f2, f3, f4 and f5 then the following would be how Kfolds cross validation:
iteration 1: validation set= f1, training set=[f2,f3,f4,f5]

iteration 2: validation set= f2, training set=[f1,f3,f4,f5]

iteration 3: validation set= f3, training set=[f1,f2,f4,f5]

iteration 4: validation set= f4, training set=[f1,f2,f3,f5]

iteration 5: validation set= f5, training set=[f1,f2,f3,f4]


Example
```
--kfold 5
--kfold_analysis_metrics MRR
--include_testset_in_kfold True
```

This would split the dataset into 5 folds: f1, f2, f3, f4 and f5 then the following would be how Kfolds cross validation:
iteration 1: validation set= f1, test set = f2 , training set=[f3,f4,f5]

iteration 2: validation set= f2, test set = f3, training set=[f1,f4,f5]

iteration 3: validation set= f3, test set = f4, training set=[f1,f2,f5]

iteration 4: validation set= f4, test set = f5, training set=[f1,f2,f3]

iteration 4: validation set= f5, test set = f1, training set=[f2,f3,f4]