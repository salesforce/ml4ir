# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.15] - 2023-01-20

### Changed

- Upgrading from tensorflow 2.0.x to 2.9.x
- Moving from Keras Functional API to Model Subclassing API for more customization capabilities
- Auxiliary loss is reimplemented as part of ScoringModel

### Added

- AutoDAGNetwork which allows for building flexible connected architectures using config files
- SetRankEncoder keras Layer to train SetRank like Ranking models
- Support for using tf-models-official deep learning garden library
- RankMatchFailure metric for validation

## [0.1.14] - 2022-11-18

### Changed

- Ability to pass custom RelevanceModel class in Pipeline.

## [0.1.13] - 2022-10-17

### Fixed

- Bug in metrics_helper when used without secondary_labels

### Added

- RankMatchFailure metric for evaluation
- RankMatchFailure auxiliary loss

## [0.1.12] - 2022-04-26

## [0.1.11] - 2021-01-18

### Changed

- Adding rank feature to serving parse fn by default and removing dependence on required serving_info attribute

## [0.1.10] - 2021-12-29

### Changed

- Adding all trained features to serving parse fn by default

## [0.1.9] - 2021-11-29

### Changed

- Refactored secondary label metrics computation for ranking and added unit tests
- Added NDCG metric for secondary labels

## [0.1.8] - 2021-10-21

### Added

- New argument to model.save()

## [0.1.7] - 2021-09-30

### Added

- SoftmaxCrossEntropy loss for ranking models

## [0.1.6] - 2021-07-16

### Fixed

- Fixing required arguments in setup.py

## [0.1.5] - 2021-07-15

### Added

- Adding support for performing post-training steps (such as copying data) by custom class inheriting RelevancePipeline.


## [0.1.4] - 2021-06-30

### Changed

- Performing pre-processing step in `__init__()` to be able to copy files before model_config and feature_config are 
  initiated.

## [0.1.3] - 2021-06-24

### Changed

- Making pyspark an optional dependency to install ml4ir

## [0.1.2] - 2021-06-16

### Added

- Support for performing pre-processing steps (such as copying data) by custom class inheriting RelevancePipeline.

## [0.1.1] - 2021-05-20

### Added

- Support for using native tf/keras feature functions from the feature config YAML

## [0.1.0] - 2021-03-01

### Changed

- TFRecord format changed for SequenceExample to earlier implementation.
- Removed support for `max_len` attribute for SequenceExample features.
- No effective changes for Example TFRecords.
- TFRecord implementation on python (training) and jvm (inference) side are now in sync.

## [0.0.5] - 2021-02-17

### Added 

- Changelog file to track version updates for ml4ir.
- `build-requirements.txt` with all python dependencies needed for developing on ml4ir and the CircleCI autobuilds.
- Updated CircleCI builds to use `build-requirements.txt`

### Fixed

- Removed build requirements from the base ml4ir `requirements.txt` allowing us to keep the published whl file dependencies to be minimal.