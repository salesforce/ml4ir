package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.ExampleBuilder;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapExampleBuilder;
import org.junit.Test;
import org.tensorflow.example.Example;

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

    private final String configFile = "feature_config.yaml";

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
                        ImmutableMap.of("query", String::toLowerCase)
                );

        String queryText = "The quick brown!";
        String domainId = "X";
        String userContext = "DDD,BBB,AAA";
        String entityId = "AAA";

        Map<String, String> queryContext =
                ImmutableMap.of("query", queryText,
                        "domain_id", domainId,
                        "user_context", userContext);

        Example example = exampleBuilder.apply(queryContext);

        ByteString queryBytesByteString = example.getFeatures().getFeatureMap().get("query_bytes").getBytesList().getValue(0);
        assertEquals(queryText.toLowerCase(), queryBytesByteString.toString("UTF-8"));

        ByteString queryWordsByteString = example.getFeatures().getFeatureMap().get("query_words").getBytesList().getValue(0);
        assertEquals(queryText.toLowerCase(), queryWordsByteString.toString("UTF-8"));
    }

}
