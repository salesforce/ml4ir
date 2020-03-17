package ml4ir.inference.tensorflow;


import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.utils.FeatureConfig;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SequenceExampleJavaBuilderTest {

    @Test
    public void testSimpleSequenceExample() {
        String query = "a query string";
        SequenceExampleJavaBuilder helper = new SequenceExampleJavaBuilder(
                FeatureConfig.apply(), "", null, null, ImmutableMap.of("query_text", query));
        SequenceExample sequenceExample = helper
                .addDoc("doc1",
                        ImmutableMap.of("f1", 1f, "f2", 0.5f),
                        ImmutableMap.of(),
                        ImmutableMap.of())
                .addDoc("doc2",
                        ImmutableMap.of("f1", 0.9f, "f2", 0.2f),
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
                floatFeature1.toArray(new Float[0]), new Float[] { 1f, 0.9f });
        List<Float> floatFeature2 =
                featureListMap.get("f2").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature f2 should match",
                floatFeature2.toArray(new Float[0]), new Float[] { 0.5f, 0.2f });
    }
}
