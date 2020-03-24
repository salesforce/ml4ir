package ml4ir.inference.tensorflow.utils;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class used for parsing the model config yaml file
 */
public class ModelFeatures {

    @JsonIgnore
    private Map<String, Map<String, String>> dataHolder = new HashMap<>();

    @JsonProperty("query_key")
    private QueryKey queryKey;

    @JsonProperty("rank")
    private Rank rank;

    @JsonProperty("label")
    private Label label;

    @JsonProperty("features")
    private List<InputFeatures> features;

    public ModelFeatures() {
    }

    public QueryKey getQueryKey() {
        return queryKey;
    }

    public Rank getRank() {
        return rank;
    }

    public Label getLabel() {
        return label;
    }

    public List<InputFeatures> getFeatures() {
        return features;
    }
}

