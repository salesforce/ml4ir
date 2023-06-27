from ml4ir.applications.ranking.model.metrics.helpers.metrics_helper import RankingConstants

class EvalConfigConstants:
    # Eval config constants
    MODE = "mode"
    BASIC_MODE = "basic"  # Just runs keras.Model.evaluate()
    EXTENDED_MODE = "extended"  # Runs the full evaluation function configured in the RelevanceModel
    GROUP_BY = "group_by"
    POWER_ANALYSIS = "power_analysis"
    VARIANCE_LIST = "variance_list"
    POWER = "power"
    PVALUE = "pvalue"
    METRICS = "metrics"


def prepare_eval_config_for_ranking(eval_config, group_key):
    """
    Reads the evaluation config yaml file and convert it into an eval config dict

    Parameters
    ----------
    evaluation_config_path : Dict
        dictionary for eval config parameters
    group_key : List[str]
        list of keys used in metric aggregation

    Returns
    -------
    eval_dict: Dict
        evaluation config yaml converted into a dict
    """
    eval_dict = {}
    eval_dict[EvalConfigConstants.MODE] = eval_config.get(EvalConfigConstants.MODE, EvalConfigConstants.EXTENDED_MODE)

    if len(eval_config) != 0:
        if EvalConfigConstants.GROUP_BY in eval_config[EvalConfigConstants.POWER_ANALYSIS] and \
                eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.GROUP_BY]:
            eval_dict[EvalConfigConstants.GROUP_BY] = eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.GROUP_BY].split(',')
        else:
            eval_dict[EvalConfigConstants.GROUP_BY] = group_key

        if EvalConfigConstants.METRICS in eval_config[EvalConfigConstants.POWER_ANALYSIS] and \
                eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.METRICS]:
            eval_dict[EvalConfigConstants.METRICS] = eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.METRICS].replace(" ","").split(',')
        else:
            eval_dict[EvalConfigConstants.METRICS] = []

        eval_dict[EvalConfigConstants.VARIANCE_LIST] = []
        for m in eval_dict[EvalConfigConstants.METRICS]:
            eval_dict[EvalConfigConstants.VARIANCE_LIST].append("old_" + m)
            eval_dict[EvalConfigConstants.VARIANCE_LIST].append("new_" + m)
        eval_dict[EvalConfigConstants.POWER] = float(eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.POWER])
        eval_dict[EvalConfigConstants.PVALUE] = float(eval_config[EvalConfigConstants.POWER_ANALYSIS][EvalConfigConstants.PVALUE])
    else:
        eval_dict[EvalConfigConstants.GROUP_BY] = group_key
        eval_dict[EvalConfigConstants.METRICS] = []
        eval_dict[EvalConfigConstants.VARIANCE_LIST] = []
        eval_dict[EvalConfigConstants.POWER] = None
        eval_dict[EvalConfigConstants.PVALUE] = EvalConfigConstants.TTEST_PVALUE_THRESHOLD
    return eval_dict