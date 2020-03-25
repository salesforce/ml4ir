package ml4ir.inference.tensorflow.data;

import com.fasterxml.jackson.annotation.JsonProperty;

public class BaseFeatures {
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
