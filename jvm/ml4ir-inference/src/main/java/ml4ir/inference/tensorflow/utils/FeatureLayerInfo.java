package ml4ir.inference.tensorflow.utils;

import com.fasterxml.jackson.annotation.JsonProperty;

public class FeatureLayerInfo {
    @JsonProperty("type")
    private String type;

    @JsonProperty("shape")
    private String shape;

    @JsonProperty("max_length")
    private int maxLength;
}
