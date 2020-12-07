package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.ExampleBuilder;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapExampleBuilder;
import org.junit.Test;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Test for the scala class {@see StringMapExampleBuilder}, which has "java-friendly" methods we need to test from
 * java-land.
 */
public class ExampleJavaBuilderTest {
    private final ClassLoader classLoader = getClass().getClassLoader();

    private final String pathFor(String name) {
        return classLoader.getResource("classification/" + name).getPath();
    }

    private final String configFile = "feature_config_with_same_name.yaml";

    /**
     * Verify that {@see Example}-building applies the correct in-jvm feature-preprocessing (lowercasing, in this
     * example) in the correctly-formed protobuf
     * @throws Exception
     */
    @Test
    public void buildProtoFromStringMap() throws Exception {
        String configPath = pathFor(configFile);
        ModelFeaturesConfig modelFeatures = ModelFeaturesConfig.load(configPath);

        ExampleBuilder<Map<String, String>> exampleBuilder =
                StringMapExampleBuilder.withFeatureProcessors(
                        modelFeatures,
                        ImmutableMap.of(),
                        ImmutableMap.of(),
                        ImmutableMap.of("query_text", String::toLowerCase)
                );

        String queryText = "The quick brown!";
        String domainId = "X";
        String userContext = "DDD,BBB,AAA";
        String entityId = "AAA";

        Map<String, String> queryContext =
                ImmutableMap.of("query_text", queryText,
                        "domain_id", domainId,
                        "user_context", userContext);

        Example example = exampleBuilder.apply(queryContext);

        Map<String, Feature> featureMap = example.getFeatures().getFeatureMap();
        ByteString queryTextByteString = featureMap.get("query_text").getBytesList().getValue(0);
        assertEquals(queryText.toLowerCase(), queryTextByteString.toString("UTF-8"));
    }

}
