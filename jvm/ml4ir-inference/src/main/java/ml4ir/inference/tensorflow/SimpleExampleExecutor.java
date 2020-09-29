package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapExampleBuilder;
import org.tensorflow.example.Example;
import java.util.Map;

public class SimpleExampleExecutor {
    private final ExampleExecutor executor;
    private final StringMapExampleBuilder protoBuilder;
    public SimpleExampleExecutor(String modelBundlePath,
                                 String inputNodeName,
                                 String outputNodeName,
                                 String featureConfigPath) {
        executor = new ExampleExecutor(
                modelBundlePath,
                ModelExecutorConfig.apply(inputNodeName, outputNodeName)
        );
        protoBuilder = StringMapExampleBuilder.withFeatureProcessors(
                ModelFeaturesConfig.load(featureConfigPath),
                ImmutableMap.of(),
                ImmutableMap.of(),
                ImmutableMap.of()
        );
    }

    public float[] predict(Map<String, String> features) {
        Example example = protoBuilder.apply(features);
        float[] predictions = executor.apply(example);
        return predictions;
    }
}
