class Metric:
    """Metrics that are populated by metrics.helpers"""
    # Primary metrics
    MRR = "MRR"
    ACR = "ACR"

    # Failure metrics on aux label
    AUX_ALL_FAILURE = "AuxAllFailure"
    AUX_INTRINSIC_FAILURE = "AuxIntrinsicFailure"  # 1 - NDCG on the aux_label

    @staticmethod
    def get_positive_metrics():
        """Metrics where higher value is better/desirable"""
        return [
            Metric.MRR
        ]

    @staticmethod
    def get_negative_metrics():
        """Metrics where lower value is better/desirable"""
        return [
            Metric.ACR,
            Metric.AUX_ALL_FAILURE,
            Metric.AUX_INTRINSIC_FAILURE
        ]
