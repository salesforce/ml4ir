package ml4ir.inference.tensorflow;

import ml4ir.inference.tensorflow.utils.ModelFeatures;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit test to verify model config yaml parsing
 */
public class ModelFeaturesParserTest {

    @Test
    public void testModelFeatureParsing() throws  Exception {
        ModelFeaturesParser modelFeaturesParser = new ModelFeaturesParser(getClass().getClassLoader().getResource("model_features.yaml").getPath());
        modelFeaturesParser.parseYaml();

        ModelFeatures modelFeatures = modelFeaturesParser.getModelFeatures();

        assertNotNull(modelFeatures);

        // Check if we parse the top level fields correctly
        assertNotNull(modelFeatures.getLabel());
        assertNotNull(modelFeatures.getQueryKey());
        assertNotNull(modelFeatures.getRank());
        assertNotNull(modelFeatures.getFeatures());

        // We have defined 12 features
        assertEquals("Unexpected number of features", 12 , modelFeatures.getFeatures().size());

    }
}
