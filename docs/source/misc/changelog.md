# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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