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

class BaseFeatures {
    @JsonProperty("name")
    private String name;

    @JsonProperty("node_name")
    private String nodeName;

    @JsonProperty("trainable")
    private boolean isTrainable;

    @JsonProperty("dtype")
    private String dtype;

    @JsonProperty("log_at_inference")
    private boolean logAtInference;

    @JsonProperty("feature_layer_info")
    private FeatureLayerInfo featureLayerInfo;

    @JsonProperty("serving_info")
    private ServingInfo servingInfo;

    @JsonProperty("tfrecord_type")
    private String tfRecordType;

    @JsonProperty("default_value")
    private String defaultValue;

    public String getNodeName() {
        return nodeName;
    }

    public String getDtype() {
        return dtype;
    }

    public ServingInfo getServingInfo() {
        return servingInfo;
    }

    public String getTfRecordType() {
        return tfRecordType;
    }

    public String getDefaultValue() {
        return defaultValue;
    }
}

class FeatureLayerInfo {
    @JsonProperty("type")
    private String type;

    @JsonProperty("shape")
    private String shape;

    @JsonProperty("max_length")
    private int maxLength;
}

class ServingInfo {
    @JsonProperty("name")
    private String name;

    @JsonProperty("required")
    private boolean isRequired;

    public String getName() {
        return name;
    }
}

class QueryKey extends BaseFeatures {}

class Rank extends BaseFeatures {}

class Label extends BaseFeatures {}

class InputFeatures extends BaseFeatures {}