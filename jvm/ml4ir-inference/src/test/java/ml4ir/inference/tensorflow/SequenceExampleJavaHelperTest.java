package ml4ir.inference.tensorflow;


import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import org.junit.Test;

import com.google.common.collect.Maps;
import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SequenceExampleJavaHelperTest {

    @Test
    public void testSimpleSequenceExample() {
        SequenceExampleJavaHelper helper = new SequenceExampleJavaHelper("a query string");
        SequenceExample sequenceExample = helper
                .addDoc("doc1", ImmutableMap.of("f1", 1f, "f2", 0.5f))
                .addDoc("doc2", ImmutableMap.of("f1", 0.9f, "f2", 0.2f))
                .build();
        ByteString queryTextByteString =
                sequenceExample.getContext().getFeatureMap().get("query_text").getBytesList().getValue(0);
        assertEquals(ByteString.copyFrom("a query string".getBytes()), queryTextByteString);
        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();
        List<Float> floatFeature1 =
                featureListMap.get("f1").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("", floatFeature1.toArray(new Float[0]), new Float[] { 1f, 0.9f });
        List<Float> floatFeature2 =
                featureListMap.get("f2").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("", floatFeature2.toArray(new Float[0]), new Float[] { 0.5f, 0.2f });
    }
}
