package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.*;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.*;

public class SequenceExampleJavaBuilderTest {

    @Test
    public void buildProtoFromStringMaps() throws Exception {
        String query = "a query string";
        Map<String, String> contextMap = ImmutableMap.of(
                "q", query,
                "query_key", "query1234",
                "userId", "john.smith@example.com");
        List<Map<String, String>> documents = ImmutableList.of(
                ImmutableMap.of(
                        "sequence_float_1", "0.1",
                        "docPopularity", "1.1",
                        "docTitle", "The title",
                        "docAgeHours", "1200",
                        "sequence_string_2", "a"),
                ImmutableMap.of(
                        "sequence_float_1", "0.2",
                        "docPopularity", "2.2",
                        "docTitle", "<html><head><title>A webpage title</title><head><title>",
                        "docAgeHours", "240"),
                ImmutableMap.of("fake", "blah", "ff2", "0.3"),
                ImmutableMap.of()
        );
        String configPath = getClass().getClassLoader().getResource("model_features.yaml").getPath();
        ModelFeaturesConfig modelFeatures = ModelFeaturesConfig.load(configPath);

        Function<Float, Float> fn = (Float count) -> (float)Math.log(1.0 + count);
        SequenceExampleBuilder<Map<String, String>, Map<String, String>> sequenceExampleBuilder =
                StringMapSequenceExampleBuilder.withFeatureProcessors(
                        modelFeatures,
                        ImmutableMap.of(
                                "docAgeHours", (Float secs) -> secs / 3600,
                                "docPopularity", (Float count) -> (float)Math.log(1.0 + count)
                        ),
                        ImmutableMap.of(), // no Long processing for any field
                        ImmutableMap.of("docTitle", String::toLowerCase)
                );


        SequenceExample sequenceExample = sequenceExampleBuilder.build(contextMap, documents);

        assertNotNull(sequenceExample);

        ByteString queryTextByteString =
                sequenceExample.getContext().getFeatureMap().get("query_text").getBytesList().getValue(0);
        assertEquals(ByteString.copyFrom(query.getBytes()), queryTextByteString);
        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();
        List<Float> floatFeature1 =
                featureListMap.get("doc_popularity").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature doc_popularity should match",
                floatFeature1.toArray(new Float[0]), new Float[] { fn.apply(1.1f), fn.apply(2.2f), fn.apply(0f), fn.apply(0f) });
        List<Float> floatFeature2 =
                featureListMap.get("doc_age_in_hours").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature doc_age_in_hours should match",
                floatFeature2.toArray(new Float[0]), new Float[] { 1200f / 3600, 240f / 3600, 2400f / 3600, 2400f / 3600 });
        List<ByteString> floatFeature3 =
                featureListMap.get("doc_title").getFeatureList().get(0).getBytesList().getValueList();
        assertArrayEquals("feature f3 should match",
                floatFeature3.toArray(new ByteString[0]), new ByteString[] {
                        ByteString.copyFromUtf8("the title"),
                        ByteString.copyFromUtf8("<html><head><title>a webpage title</title><head><title>"),
                        ByteString.copyFromUtf8(""),
                        ByteString.copyFromUtf8("")
        });
        assertNull(featureListMap.get("fake_feat"));
    }

    @Test
    public void testStringMapFeaturePreprocessing() throws Exception {
        ModelFeaturesConfig modelFeatures = ModelFeaturesConfig.load(getClass().getClassLoader().getResource("model_features.yaml").getPath());
        FeaturePreprocessor<Map<String, String>> processor = FeatureProcessors.forStringMaps(
                modelFeatures,
                "sequence",
                ImmutableMap.of(
                        "docAgeHours", (Float secs) -> secs / 3600,
                        "docPopularity", (Float count) -> (float) Math.log(count)
                ),
                ImmutableMap.of(), // no Long processing for any field
                ImmutableMap.of("docTitle", String::toLowerCase));
        Example example = processor.apply(ImmutableMap.of(
                "sequence_float_1", "0.1",
                "sequence_float_2", "1.1",
                "docTitle", "The title",
                "docAgeHours", "1200",
                "sequence_string_2", "A string!"));
        Object docAgeHours = example.features().floatFeatures().get("doc_age_in_hours").get();
        assertEquals( 1200.0 / 3600 , (float) docAgeHours, 0.001f);
        Object sequenceLong2 = example.features().int64Features().get("sequence_long_2").get();
        assertEquals(2147483649L, (long) sequenceLong2);
        String docTitle = example.features().stringFeatures().get("doc_title").get();
        assertEquals("the title", docTitle);
    }
}
