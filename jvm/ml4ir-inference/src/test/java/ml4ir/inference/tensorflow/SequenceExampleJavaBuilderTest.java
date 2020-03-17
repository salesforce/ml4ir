package ml4ir.inference.tensorflow;


import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.utils.FeatureConfig;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class SequenceExampleJavaBuilderTest {

    @Test
    public void testSimpleSequenceExample() {
        String query = "a query string";

        // TODO: this should encode that f1, f2, and f3 are in the model and have default values, but fake_feat is not
        FeatureConfig featureConfig = FeatureConfig.apply();

        SequenceExampleJavaBuilder helper = new SequenceExampleJavaBuilder(
                featureConfig,
                "ignored",
                ImmutableMap.of(),
                ImmutableMap.of(),
                ImmutableMap.of("query_text", query));
        SequenceExample sequenceExample = helper
                .addDoc("doc1",
                        ImmutableMap.of("f1", 1f, "f2", 0.1f, "f3", 0.01f),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .addDoc("doc2",
                        ImmutableMap.of("f1", 2f, "f2", 0.2f, "f3", 0.02f),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .addDoc("doc3",
                        ImmutableMap.of(/* no f1 -> anything */"f2", 0.3f, "f3", 0.03f),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .addDoc("doc4",
                        ImmutableMap.of("f1", 4f, "fake_feat", -1f /* no f2 or f3 */),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .addDoc("doc5",
                        ImmutableMap.of("f1", 5f, "f2", 0.5f, "f3", 0.05f),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .build();
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
                floatFeature3.toArray(new Float[0]), new Float[] { 0.01f, 0.02f, 0.03f, 0f, 0.5f });
        assertNull(featureListMap.get("fake_feat"));
    }
}
