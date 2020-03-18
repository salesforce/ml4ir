package ml4ir.inference.tensorflow;

import ml4ir.inference.tensorflow.utils.FeatureConfig;
import ml4ir.inference.tensorflow.utils.FeatureField;
import org.tensorflow.DataType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class FeatureConfigJavaBuilder {

    private List<FeatureField> contextFeatures;
    private List<FeatureField> sequenceFeatures;

    public FeatureConfigJavaBuilder() {
        this.contextFeatures = new ArrayList<>();
        this.sequenceFeatures = new ArrayList<>();
    }

    public FeatureConfigJavaBuilder addContextFeatures(String name, String dataType) {
        this.contextFeatures.add(new FeatureField(name, DataType.valueOf(dataType.toUpperCase())));
        return this;
    }

    public FeatureConfigJavaBuilder addContextFeaturesBulk(Map<String, String> contextMetadata) {
        contextMetadata.forEach((k,v) -> this.contextFeatures.add(
                new FeatureField(k, DataType.valueOf(v.toUpperCase()))));
        return this;
    }

    public FeatureConfigJavaBuilder addSequenceFeatures(String name, String dataType) {
        this.sequenceFeatures.add(new FeatureField(name, DataType.valueOf(dataType.toUpperCase())));
        return this;
    }

    public FeatureConfigJavaBuilder addSequenceFeaturesBulk(Map<String, String> sequenceMetadata) {
        sequenceMetadata.forEach((k,v) -> this.sequenceFeatures.add(
                new FeatureField(k, DataType.valueOf(v.toUpperCase()))));
        return this;
    }

    public FeatureConfig build() {
        return FeatureConfig.apply(contextFeatures, sequenceFeatures);
    }
}
