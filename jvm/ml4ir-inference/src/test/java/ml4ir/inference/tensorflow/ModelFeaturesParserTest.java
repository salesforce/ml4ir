package ml4ir.inference.tensorflow;

import ml4ir.inference.tensorflow.utils.ModelFeatures;
import org.junit.Test;
import org.tensorflow.DataType;

import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class ModelFeaturesParserTest {

    @Test
    public void testModelFeatureParsing() throws Exception {
        ModelFeaturesParser modelFeaturesParser = new ModelFeaturesParser(getClass().getClassLoader().getResource("model_features.yaml").getPath());
        modelFeaturesParser.parseYaml();

        ModelFeatures modelFeatures =modelFeaturesParser.getModelFeatures();

        assertNotNull(modelFeatures);

        // We defined 12 features
        assertEquals(12 , modelFeatures.getFeatures().size());

        Map<String, String> contextDataTypes = modelFeaturesParser.getContextDataTypes();
        assertEquals("float" , contextDataTypes.get("context_float_1"));
        assertEquals("string" , contextDataTypes.get("query_key"));

        Map<String, String> sequenceDataTypes = modelFeaturesParser.getSequenceDataTypes();
        assertEquals("string" , sequenceDataTypes.get("sequence_string_1"));
        assertEquals("float" , sequenceDataTypes.get("rank"));

        Map<String, String> contextDefaultVal = modelFeaturesParser.getContextFeaturesForDataType(DataType.FLOAT);
        assertEquals(0f, Float.valueOf(contextDefaultVal.get("context_float_1")), 0.01);
        assertEquals(1.5f, Float.valueOf(contextDefaultVal.get("context_float_2")), 0.01);

        Map<String, String> sequenceDefaultValString = modelFeaturesParser.getSequenceFeaturesForDataType(DataType.STRING);
        assertEquals("", sequenceDefaultValString.get("sequence_string_1"));
        assertEquals("sequence_string_2", sequenceDefaultValString.get("sequence_string_2"));

        Map<String, String> sequenceDefaultValLong = modelFeaturesParser.getSequenceFeaturesForDataType(DataType.INT64);
        assertEquals(0L, (long) Long.valueOf(sequenceDefaultValLong.get("sequence_long_1")));
        assertEquals(2147483649L, (long) Long.valueOf(sequenceDefaultValLong.get("sequence_long_2")));

        Map<String, String> contextServingName = modelFeaturesParser.getServingNameMappingForContextFeatures();
        assertEquals("query_key_serve", contextServingName.get("query_key"));

        Map<String, String> sequenceServingName = modelFeaturesParser.getServingNameMappingForSequenceFeatures();
        assertEquals("rank_serve", sequenceServingName.get("rank"));

    }
}
