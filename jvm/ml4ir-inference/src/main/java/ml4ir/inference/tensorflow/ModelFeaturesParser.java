package ml4ir.inference.tensorflow;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import ml4ir.inference.tensorflow.data.ModelFeatures;

import java.io.File;
import java.io.IOException;

/**
 * Parse the input yaml file to extract the features
 */
public class ModelFeaturesParser {

    private String path;
    private ModelFeatures modelFeatures;

    public ModelFeaturesParser(String yamlPath) {
        this.path = yamlPath;
    }

    public void parseYaml() throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.modelFeatures = mapper.readValue(new File(this.path), ModelFeatures.class);
    }

    public static ModelFeatures parseModelFeaturesConfig(String yamlPath) throws IOException {
        ModelFeaturesParser parser = new ModelFeaturesParser(yamlPath);
        parser.parseYaml();
        return parser.getModelFeatures();
    }

    ModelFeatures getModelFeatures() {
        return this.modelFeatures;
    }
}
