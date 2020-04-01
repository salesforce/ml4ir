package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import ml4ir.inference.tensorflow.data.*;
import org.junit.Test;

import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;
import scala.Function1;

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
                        "context_float_1", "0.1",
                        "context_float_2", "1.1",
                        "docTitle", "The title",
                        "sequence_string_2", "a"),
                ImmutableMap.of(
                        "context_float_1", "0.2",
                        "docPopularity", "2.2",
                        "docTitle", "<html><head><title>A webpage title</title><head><title>",
                        "doc_age_in_hours", "240"),
                ImmutableMap.of("fake", "blah", "ff2", "0.3"),
                ImmutableMap.of()
        );

        ModelFeatures modelFeatures = ModelFeaturesParser.parseModelFeaturesConfig(
                getClass().getClassLoader().getResource("model_features.yaml").getPath());

        Function<? super Float, ? extends Float> s = (Float secs) -> secs / 3600;
        Function<? super Float, ? extends Float> l = (Float count) -> (float)Math.log(count);
        Function<? super String, ? extends String> lc = String::toLowerCase;
        Map<String, Function<Float, Float>> floatFns = ImmutableMap.of(
                "docAgeHours", ((Float secs) -> secs / 3600),
                "docPopularity", ((Float count) -> (float)Math.log(count))
        );
        FeaturePreprocessor<Map<String, String>> ctxPre =
                new StringMapFeatureProcessor(modelFeatures,
                        "context");
        FeaturePreprocessor<Map<String, String>> seqPre =
                new StringMapFeatureProcessor(modelFeatures, "sequence", new PrimitiveProcessor() {
                    @Override
                    public float processFloat(String servingName, float f) {
                        return floatFns.getOrDefault(servingName, Function.identity()).apply(f);
                    }
                });
        PrimitiveProcessor pp = PrimitiveProcessors.fromJavaFloatFunction()

        StringMapSequenceExampleBuilder sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(
                        modelFeatures, new PrimitiveProcessor() {
                            @Override
                            public String processString(String servingName, String s) {
                                return s.toLowerCase();
                            }
                            @Override
                            public float processFloat(String servingName, float f) {
                                return floatFns.getOrDefault(servingName, Function.identity()).apply(f);
                            }
                        }
                        /*ImmutableMap.of(
                                "docAgeHours", func((Float secs) -> secs / 3600),
                                "docPopularity", func((Float count) -> (float)Math.log(count))
                        ),
                        ImmutableMap.of(), // no Long processing for any field
                        ImmutableMap.of(
                                "docTitle", func((String str) -> str.toLowerCase())
                        )*/
                );


        SequenceExample sequenceExample = sequenceExampleBuilder.build(contextMap, documents);

        assertNotNull(sequenceExample);

        // FIXME: this is currently failing, because it's not checking the right things.  Need to reconcile the
        // FIXME: fake data with the yaml config and the output expectations
        ByteString queryTextByteString =
                sequenceExample.getContext().getFeatureMap().get("query_text").getBytesList().getValue(0);
        assertEquals(ByteString.copyFrom(query.getBytes()), queryTextByteString);
        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();
        List<Float> floatFeature1 =
                featureListMap.get("context_float_1").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature context_float_1 should match",
                floatFeature1.toArray(new Float[0]), new Float[] { 0.1f, 0.2f, 0f, 0f });
        List<Float> floatFeature2 =
                featureListMap.get("doc_age_in_hours").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature doc_age_in_hours should match",
                floatFeature2.toArray(new Float[0]), new Float[] { 0.1f, 0.2f, 0.3f, 0f, 0.5f });
        List<Float> floatFeature3 =
                featureListMap.get("f3").getFeatureList().get(0).getFloatList().getValueList();
        assertArrayEquals("feature f3 should match",
                floatFeature3.toArray(new Float[0]), new Float[] { 0.01f, 0.02f, 0.03f, 0f, 0.05f });
        assertNull(featureListMap.get("fake_feat"));
    }
}
