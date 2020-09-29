package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapSequenceExampleBuilder;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

public class SimpleSequenceExampleExecutor {
    private final SequenceExampleExecutor executor;
    private final StringMapSequenceExampleBuilder protoBuilder;
    public SimpleSequenceExampleExecutor(String modelBundlePath,
                                         String inputNodeName,
                                         String outputNodeName,
                                         String featureConfigPath) {
        executor = new SequenceExampleExecutor(
                modelBundlePath,
                ModelExecutorConfig.apply(inputNodeName, outputNodeName)
        );
        protoBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(
                ModelFeaturesConfig.load(featureConfigPath),
                ImmutableMap.of(),
                ImmutableMap.of(),
                ImmutableMap.of()
        );
    }

    public float[] predict(Map<String, String> contextFeatures, List<Map<String, String>> recordFeatures) {
        SequenceExample sequenceExample = protoBuilder.build(contextFeatures, recordFeatures);
        float[] predictions = executor.apply(sequenceExample);
        return predictions;
    }
}
