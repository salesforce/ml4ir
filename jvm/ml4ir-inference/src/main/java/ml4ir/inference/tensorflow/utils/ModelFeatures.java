package ml4ir.inference.tensorflow.utils;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableMap;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

    public ModelFeatures() {}

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

    public Map<String, Map<String, String>> getDataTypesForFeatures() {
        dataHolder.clear();
        buildMapHelper(dataHolder, this.getLabel().getTfRecordType(),
                this.getLabel().getName(), this.getLabel().getDtype());
        buildMapHelper(dataHolder, this.getRank().getTfRecordType(),
                this.getRank().getName(), this.getRank().getDtype());
        buildMapHelper(dataHolder, this.getQueryKey().getTfRecordType(),
                this.getQueryKey().getName(), this.getQueryKey().getDtype());
        features.forEach(f -> buildMapHelper(dataHolder, f.getTfRecordType(),
                f.getName(), f.getDtype()));
        return dataHolder;
    }

    public  Map<String, Map<String, String>> getDefaultValuesForFeatures() {
        dataHolder.clear();
        buildMapHelper(dataHolder, this.getLabel().getTfRecordType(),
                this.getLabel().getName(), this.getLabel().getDefaultValue());
        buildMapHelper(dataHolder, this.getRank().getTfRecordType(),
                this.getRank().getName(), this.getRank().getDefaultValue());
        buildMapHelper(dataHolder, this.getQueryKey().getTfRecordType(),
                this.getQueryKey().getName(), this.getQueryKey().getDefaultValue());
        features.forEach(f -> buildMapHelper(dataHolder, f.getTfRecordType(),
                f.getName(), f.getDefaultValue()));
        return dataHolder;
    }

    public Map<String, Map<String, String>> getServingNameMappingForFeatures() {
        dataHolder.clear();
        buildMapHelper(dataHolder, this.getLabel().getTfRecordType(),
                this.getLabel().getName(), this.getLabel().getServingInfo().getName());
        buildMapHelper(dataHolder, this.getRank().getTfRecordType(),
                this.getRank().getName(), this.getRank().getServingInfo().getName());
        buildMapHelper(dataHolder, this.getQueryKey().getTfRecordType(),
                this.getQueryKey().getName(), this.getQueryKey().getServingInfo().getName());
        features.forEach(f -> buildMapHelper(dataHolder, f.getTfRecordType(),
                f.getName(), f.getServingInfo().getName()));
        return dataHolder;
    }

    public void buildMapHelper(Map<String, Map<String, String>> data,
                               String s1, String s2, String s3) {
        Map<String, String> innerMap = data.getOrDefault(s1, new HashMap<>());
        innerMap.put(s2, s3);
        data.put(s1, innerMap);
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

    public String getName() {
        return name;
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