package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.*;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.io.FileOutputStream;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.*;

public class SequenceExampleJavaBuilderTest {
    private final ClassLoader classLoader = getClass().getClassLoader();
    private String configFile = "model_features_0_0_2.yaml";

    @Test
    public void buildProtoFromStringMaps() throws Exception {
        String query = "A query string";
        Map<String, String> contextMap = ImmutableMap.of(
                "q", query,
                "query_key", "query1234",
                "userId", "john.smith@example.com");
        List<Map<String, String>> documents = ImmutableList.of(
                ImmutableMap.of(
                        "floatFeat0", "0.1",
                        "floatFeat1", "0.2",
                        "floatFeat2", "0.3"),
                ImmutableMap.of(
                        "floatFeat0", "1.1",
                        "floatFeat1", "1.2",
                        "floatFeat2", "1.3"),
                ImmutableMap.of("fake", "blah", "ff2", "0.3"),
                ImmutableMap.of()
        );
        String configPath = classLoader.getResource(configFile).getPath();
        ModelFeaturesConfig modelFeatures = ModelFeaturesConfig.load(configPath);

        Function<Float, Float> log1p = (Float count) -> (float)Math.log(1.0 + count);
        SequenceExampleBuilder<Map<String, String>, Map<String, String>> sequenceExampleBuilder =
                StringMapSequenceExampleBuilder.withFeatureProcessors(
                        modelFeatures,
                        ImmutableMap.of(
                                "floatFeat2", (Float f) -> f * 10,
                                "floatFeat1", log1p
                        ),
                        ImmutableMap.of(), // no Long processing for any field
                        ImmutableMap.of("q", String::toLowerCase)
                );


        SequenceExample sequenceExample = sequenceExampleBuilder.build(contextMap, documents);

        assertNotNull(sequenceExample);

        ByteString queryTextByteString =
                sequenceExample.getContext().getFeatureMap().get("query_str").getBytesList().getValue(0);
        assertEquals(ByteString.copyFrom(query.toLowerCase().getBytes()), queryTextByteString);

        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();
        List<Float> floatFeature1 =
                featureListMap.get("feat_1").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals(
                "feat_1 should match inputs from floatFeat1",
                new Float[] { log1p.apply(0.2f), log1p.apply(1.2f), log1p.apply(0f), log1p.apply(0f) },
                floatFeature1.toArray(new Float[0])
        );
        List<Float> floatFeature2 =
                featureListMap.get("feat_2").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals(
                "feat_0 should match inputs from docAgeHours",
                new Float[] { 3.0f, 13f, 0f, 0f },
                floatFeature2.toArray(new Float[0])
        );
        assertNull(featureListMap.get("fake_feat"));
    }
}
