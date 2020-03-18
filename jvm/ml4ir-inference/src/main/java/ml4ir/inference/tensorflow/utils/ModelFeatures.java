package ml4ir.inference.tensorflow.utils;

import com.google.common.collect.ImmutableMap;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelFeatures {
    private QueryKey query_key;
    private Rank rank;
    private Label label;
    private List<InputFeatures> features;

    public ModelFeatures() {}

    public QueryKey getQuery_key() {
        return query_key;
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
        Map<String, Map<String, String>> data = new HashMap<>();
        data.put(this.getLabel().getTfrecord_type(),
                ImmutableMap.of(this.getLabel().getName(), this.getLabel().getDtype()));
        data.put(this.getRank().getTfrecord_type(),
                ImmutableMap.of(this.getRank().getName(), this.getRank().getDtype()));
        data.put(this.getQuery_key().getTfrecord_type(),
                ImmutableMap.of(this.getQuery_key().getName(), this.getLabel().getDtype()));
        features.forEach(f -> data.put(f.getTfrecord_type(),
                ImmutableMap.of(f.getName(), f.getDtype())));
        return data;
    }

    public  Map<String, Map<String, String>> getDefaultValuesForFeatures() {
        Map<String, Map<String, String>> data = new HashMap<>();
        data.put(this.getLabel().getTfrecord_type(),
                ImmutableMap.of(this.getLabel().getName(), this.getLabel().getDefault_value()));
        data.put(this.getRank().getTfrecord_type(),
                ImmutableMap.of(this.getRank().getName(), this.getRank().getDefault_value()));
        data.put(this.getQuery_key().getTfrecord_type(),
                ImmutableMap.of(this.getQuery_key().getName(), this.getQuery_key().getDefault_value()));
        features.forEach(f -> data.put(f.getTfrecord_type(),
                ImmutableMap.of(f.getName(), f.getDefault_value())));
        return data;
    }

    public Map<String, Map<String, String>> getServingNameMappingForFeatures() {
        Map<String, Map<String, String>> data = new HashMap<>();
        data.put(this.getLabel().getTfrecord_type(),
                ImmutableMap.of(this.getLabel().getName(), this.getLabel().getServing_info().getName()));
        data.put(this.getRank().getTfrecord_type(),
                ImmutableMap.of(this.getRank().getName(), this.getRank().getServing_info().getName()));
        data.put(this.getQuery_key().getTfrecord_type(),
                ImmutableMap.of(this.getQuery_key().getName(), this.getQuery_key().getServing_info().getName()));
        features.forEach(f -> data.put(f.getTfrecord_type(),
                ImmutableMap.of(f.getName(), f.getServing_info().getName())));
        return data;
    }
}

class BaseFeatures {
    private String name;
    private String node_name;
    private boolean trainable;
    private String dtype;
    private boolean log_at_inference;
    private FeatureLayerInfo feature_layer_info;
    private ServingInfo serving_info;
    private String tfrecord_type;
    private String default_value;

    public String getName() {
        return name;
    }

    public String getNode_name() {
        return node_name;
    }

    public boolean isTrainable() {
        return trainable;
    }

    public String getDtype() {
        return dtype;
    }

    public boolean isLog_at_inference() {
        return log_at_inference;
    }

    public FeatureLayerInfo getFeature_layer_info() {
        return feature_layer_info;
    }

    public ServingInfo getServing_info() {
        return serving_info;
    }

    public String getTfrecord_type() {
        return tfrecord_type;
    }

    public String getDefault_value() {
        return default_value;
    }
}

class FeatureLayerInfo {
    private String type;
    private String shape;
    private int max_length;

    public String getType() {
        return type;
    }

    public String getShape() {
        return shape;
    }

    public int getMax_length() {
        return max_length;
    }
}

class ServingInfo {
    private String name;
    private boolean required;

    public String getName() {
        return name;
    }

    public boolean isRequired() {
        return required;
    }
}

class QueryKey extends BaseFeatures {}

class Rank extends BaseFeatures {}

class Label extends BaseFeatures {}

class InputFeatures extends BaseFeatures {}