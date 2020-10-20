## Predicting with a model trained on ml4ir

This sections explores how to get predictions from a model that is trained with `ml4ir`.
For the sake of example, we assume that we have already trained a classification model. To train such a model, see [this notebook](https://github.com/salesforce/ml4ir/blob/master/python/notebooks/EntityPredictionDemo.ipynb).

The model artifacts are as follows in the `models-dir`:
```bash
├── checkpoint.tf
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── final
    ├── default
    │   ├── assets
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── layers
    │   ├── bidirectional.npz
    │   ├── bidirectional_1.npz
    │   ├── LAYERS as npz files
    │   ├── .
    │   ├── .
    │   └── vocab_lookup_3.npz
    └── tfrecord
        ├── assets
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index
```
The `final/default` signature is used when we hit the model with tensors.
The `final/tfrecord` signature is used when we hit it with tfrecords.

### Predicting with the tfrecords signature
The second case, which is easier when our data are already in tfrecords requires:
```python
from tensorflow import data
import tensorflow as tf
from tensorflow.keras import models as kmodels

MODEL_DIR = "/PATH/TO/MODEL/"

model = kmodels.load_model(os.path.join(MODEL_DIR, 'final/tfrecord/'), compile=False)
infer_fn = model.signatures["serving_tfrecord"]
```
And now to construct a dataset and get predictions on it:
```python
dataset = data.TFRecordDataset(glob.glob(os.path.join('/PATH/TO/DATASET', "part*")))
total_preds =  []
i = 0
# A prediction loop; to predict to one batch we can simply `infer_fn(next(iter(dataset)))`
for batch in dataset.batch(1024):
    probs = infer_fn(protos=batch)
    total_preds.append(probs)
# Post processing of predictions
```

### Predicting with the default signature
The default signature requires hitting the model with tensors. This, in turn, requires to
do all the required preprocessing (look-ups, etc) to get these tensors.
This is done with ml4ir. The code sceleton below describes the required steps.

```python
# Define logger
# Define feat_config
# Define RelevanceDataset
# Defing RelevanceModel

relevance_model.predict(relevance_dataset.test)
```

This process, while much more verbose allows to do custom pre-processing on the model
inputs, which can be different from the preprocessing done during training.
For images, this can be artificial blurring. For text classification, using a subset of the
text and many others.

Recall, pre-processing in ml4ir is controlled in the feature_config.yaml file.
To do something extra during inference, we need to add it to the feature config, so that the
pipeline is updated.
For example, to use only the first few bytes of a text field called query that it is currently
only preprocessed by lower-casing it, we need a function that achieves this and to pass the details
in the config.
So before, the features config could be:
```bash
    preprocessing_info:
      - fn: preprocess_text
        args:
          remove_punctuation: true
          to_lower: true
```
so that `preprocess_text` is the only preprocessing function. We can now do
```bash
    preprocessing_info:
      - fn: preprocess_text
        args:
          remove_punctuation: true
          to_lower: true
      - fn: trim_text
        args:
          keep_first: 3
```
and define trim text in the code.
Assuming that:
```python
@tf.function
def trim_text(inp, keep_first=3):
    """Keeps the first `keep_first` bytes of a tf.string"""
    return tf.strings.substr(inp, 0, keep_first, unit='BYTE')
```
then defining the RelevanceDataset as:
```
relevance_dataset = RelevanceDataset(
        data_dir="/tmp/dataset",
        data_format=DataFormatKey.TFRECORD,
        feature_config=feature_config,
        tfrecord_type=TFRecordTypeKey.EXAMPLE,
        batch_size=1024,
        preprocessing_keys_to_fns={'trim_text': trim_text},  # IMPORTANT!
        file_io=file_io, use_part_files=True,
        logger=logger
    )
```
will result in queries whose size is 3 bytes (as described in `trim_text`).

For more information on these, [please refer to this notebook](https://github.com/salesforce/ml4ir/blob/master/python/notebooks/predicting_with_ml4ir.ipynb)
