package ml4ir.inference.tensorflow;


import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.Example;
import ml4ir.inference.tensorflow.data.StringMapFeatureProcessor;
import ml4ir.inference.tensorflow.data.ModelFeatures;
import ml4ir.inference.tensorflow.data.SequenceExampleBuilder;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class SequenceExampleJavaBuilderTest {

    @Test
    public void buildProtoFromStringMaps() throws Exception {
        String query = "a query string";
        Map<String, String> contextMap = ImmutableMap.of(
                "query_text", query,
                "query_id", "query1234",
                "user_id", "john.smith@example.com");
        List<Map<String, String>> documents = ImmutableList.of(
                ImmutableMap.of("ff1", "0.1", "ff2", "1.1", "lf1", "11L", "sf1", "a"),
                ImmutableMap.of("ff1", "0.2", "ff2", "2.2", "lf1",  "22L", "sf1", "b"),
                ImmutableMap.of("fake", "blah", "ff2", "0.3"),
                ImmutableMap.of()
        );
        ModelFeatures modelFeatures = ModelFeaturesParser.parseModelFeaturesConfig(
                getClass().getClassLoader().getResource("model_features.yaml").getPath());

        StringMapFeatureProcessor contextPreprocessor =
                new StringMapFeatureProcessor(modelFeatures, "context");
        Example contextExample = contextPreprocessor.apply(contextMap);

        StringMapFeatureProcessor examplePreprocessor =
                new StringMapFeatureProcessor(modelFeatures, "sequence");
        Example[] sequenceExamples = documents.stream().map(examplePreprocessor::apply).toArray(Example[]::new);

        SequenceExample sequenceExample = new SequenceExampleBuilder().apply(contextExample, sequenceExamples);

        assertNotNull(sequenceExample);

        // FIXME: this is currently failing, because it's not checking the right things.  Need to reconcile the
        // FIXME: fake data with the yaml config and the output expectations
        ByteString queryTextByteString =
                sequenceExample.getContext().getFeatureMap().get("query_text").getBytesList().getValue(0);
        assertEquals(ByteString.copyFrom(query.getBytes()), queryTextByteString);
        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();
        List<Float> floatFeature1 =
                featureListMap.get("f1").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature f1 should match",
                floatFeature1.toArray(new Float[0]), new Float[] { 1f, 2f, 0f, 4f, 5f });
        List<Float> floatFeature2 =
                featureListMap.get("f2").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature f2 should match",
                floatFeature2.toArray(new Float[0]), new Float[] { 0.1f, 0.2f, 0.3f, 0f, 0.5f });
        List<Float> floatFeature3 =
                featureListMap.get("f3").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature f3 should match",
                floatFeature3.toArray(new Float[0]), new Float[] { 0.01f, 0.02f, 0.03f, 0f, 0.05f });
        assertNull(featureListMap.get("fake_feat"));
    }
}
