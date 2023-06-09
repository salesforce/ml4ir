class Metric:
    """Metrics that are populated by metrics.helpers"""
    # Primary metrics
    MRR = "MRR"
    ACR = "ACR"
    NDCG = "NDCG"

    # Failure metrics on aux label
    AUX_ALL_FAILURE = "AuxAllFailure"
    AUX_INTRINSIC_FAILURE = "AuxIntrinsicFailure"  # 1 - NDCG on the aux_label
    AUX_RANKMF = "AuxRankMF"  # RankMatchFailure computed on the auxiliary label

    @staticmethod
    def get_positive_metrics():
        """Metrics where higher value is better/desirable"""
        return [
            Metric.MRR,
            Metric.NDCG
        ]

    @staticmethod
    def get_negative_metrics():
        """Metrics where lower value is better/desirable"""
        return [
            Metric.ACR,
            Metric.AUX_ALL_FAILURE,
            Metric.AUX_INTRINSIC_FAILURE,
            Metric.AUX_RANKMF
        ]

    @staticmethod
    def get_all_metrics():
        """returns all the metrics"""
        return[
            Metric.MRR,
            Metric.ACR,
            Metric.NDCG,
            Metric.AUX_ALL_FAILURE,
            Metric.AUX_INTRINSIC_FAILURE,
            Metric.AUX_RANKMF]

    @staticmethod
    def get_all_aux_metrics():
        """returns only the metrics related to auxiliary loss"""
        return [
            Metric.AUX_ALL_FAILURE,
            Metric.AUX_INTRINSIC_FAILURE,
            Metric.AUX_RANKMF]

    @staticmethod
    def get_metrics_with_new_old_prefix(metric_list):
        """
        Adds the prefix '_old', '_new' to the given metric list. This is needed for power analysis

        Parameters
        ----------
        metric_list: list
            A list of metrics

        Returns
        ----------
        new_old_list: list
            The list of metrics with the prefixes '_new', '_old' added.
        """
        new_old_list = []
        for m in metric_list:
            new_old_list.append("old_" + m)
            new_old_list.append("new_" + m)
        return new_old_list


