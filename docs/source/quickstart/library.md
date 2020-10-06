## Using ml4ir as a library

Let's try to train a simple Learning-to-Rank model with some sample data...


**Setup FileIO handler(and logger)**
```
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.file_io import FileIO
import logging

file_io : FileIO = LocalIO()

# Set up logger
logger = logging.getLogger()
```

**Load the FeatureConfig from a predefined YAML file**

More information about defining a feature configuration YAML file **[here](/quickstart/feature_config)**
```
from ml4ir.base.features.feature_config import FeatureConfig, SequenceExampleFeatureConfig
from ml4ir.base.config.keys import *

feature_config: SequenceExampleFeatureConfig = FeatureConfig.get_instance(
    tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
    feature_config_dict=file_io.read_yaml(FEATURE_CONFIG_PATH),
    logger=logger)
```

**Create a RelevanceDataset**

More information about the data loading pipeline **[here](/quickstart/dataset)**
```
from ml4ir.base.data.relevance_dataset import RelevanceDataset

relevance_dataset = RelevanceDataset(data_dir=DATA_DIR,
                            data_format=DataFormatKey.CSV,
                            feature_config=feature_config,
                            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
                            max_sequence_size=MAX_SEQUENCE_SIZE,
                            batch_size=128,
                            preprocessing_keys_to_fns={},
                            file_io=file_io,
                            logger=logger)
```

**Define an InteractionModel**
```
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel

interaction_model: InteractionModel = UnivariateInteractionModel(
                                            feature_config=feature_config,
                                            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
                                            max_sequence_size=MAX_SEQUENCE_SIZE,
                                            feature_layer_keys_to_fns={},
                                            file_io=file_io,
                                        )
```

**Define losses, metrics and optimizer**

Here, we are using predefined losses, metrics and optimizers. But each of these can be customized as needed.
```
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.applications.ranking.model.losses import loss_factory
from ml4ir.applications.ranking.model.metrics import metric_factory
from ml4ir.applications.ranking.config.keys import LossKey, MetricKey, ScoringTypeKey

# Define loss object from loss key
loss: RelevanceLossBase = loss_factory.get_loss(
                                loss_key=LossKey.RANK_ONE_LISTNET,
                                scoring_type=ScoringTypeKey.POINTWISE)
    
# Define metrics objects from metrics keys
metric_keys = [MetricKey.MRR, MetricKey.ACR]
metrics: List[Union[Type[Metric], str]] = [metric_factory.get_metric(metric_key=m) for m in metric_keys]
    
# Define optimizer
optimizer: Optimizer = get_optimizer(
                            optimizer_key=OptimizerKey.ADAM,
                            learning_rate=0.001
                        )
```

**Define the Scorer object by wrapping the InteractionModel and the loss function**
```
scorer: RelevanceScorer = RelevanceScorer.from_model_config_file(
    model_config_file=MODEL_CONFIG_PATH,
    interaction_model=interaction_model,
    loss=loss,
    logger=logger,
    file_io=file_io,
)
```

**Combine it all to create a RankingModel**
```
ranking_model: RelevanceModel = RankingModel(
                                    feature_config=feature_config,
                                    tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
                                    scorer=scorer,
                                    metrics=metrics,
                                    optimizer=optimizer,
                                    file_io=file_io,
                                    logger=logger,
                                )
```

**Training the RankingModel and monitor for MRR metric**
```
ranking_model.fit(dataset=relevance_dataset,
                  num_epochs=3, 
                  models_dir=MODELS_DIR,
                  logs_dir=LOGS_DIR,
                  monitor_metric="new_MRR",
                  monitor_mode="max")
```

**Run inference on the RankingModel**
```
ranking_model.predict(test_dataset=relevance_dataset.test).sample(10)
```

**Finally, save the model**

One can additionally pass preprocessing functions to be persisted as part of the SavedModel and into the tensorflow graph. For more information on how to do this, check **[here](/quickstart/saving)**
```
ranking_model.save(models_dir=MODELS_DIR,
                   preprocessing_keys_to_fns={},
                   required_fields_only=True)
```

For details on serving this model on the JVM check **[this guide](/quickstart/serving)**