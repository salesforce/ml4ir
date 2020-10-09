from ml4ir.applications.classification.model.metrics.metrics_impl import Top5CategoricalAccuracy


def test_top_5_categorical_accuracy():
    """Test for calculating top_5_categorical_accuracy metric"""
    metric = Top5CategoricalAccuracy(feature_config=None, metadata_features={})
    metric.update_state(
        [[[0, 0, 1, 0, 0, 0]], [[0, 1, 0, 0, 0, 0]]],
        [[[0.1, 0.1, 0.4, 0.2, 0.1, 0.1]], [[0.19, 0.01, 0.4, 0.2, 0.1, 0.1]]],
    )
    score = metric.result().numpy()
    assert score == 0.5
