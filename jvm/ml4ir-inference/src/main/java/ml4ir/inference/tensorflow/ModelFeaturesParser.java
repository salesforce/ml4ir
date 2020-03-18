package ml4ir.inference.tensorflow;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.google.common.collect.ImmutableMap;
import ml4ir.inference.tensorflow.utils.ModelFeatures;
import org.tensorflow.DataType;

import java.io.File;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * It returns a list of values
 */

public class ModelFeaturesParser {

    private static final String CONTEXT = "context";
    private static final String SEQUENCE = "sequence";

    private String path;
    private ModelFeatures modelFeatures;
    private Map<String, Map<String, String>> features;

    public ModelFeaturesParser(String yamlPath) {
        this.path = yamlPath;
    }

    public void parseYaml() throws Exception {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.modelFeatures = mapper.readValue(new File(this.path), ModelFeatures.class);
    }

    public Map<String, String> getContextDataTypes() {
        return modelFeatures.getDataTypesForFeatures().get(CONTEXT);
    }

    public Map<String, String> getSequenceDataTypes() {
        return modelFeatures.getDataTypesForFeatures().get(SEQUENCE);
    }

    /**
     * This will return a map of <feature_name, default_value> for a given data type
     * for tfRecord type Context
     */
    public <T> Map<String, T> getContextFeaturesForDataType(DataType dtype) {
        return defaultValueExtractor(modelFeatures.getDataTypesForFeatures().get(CONTEXT),
                modelFeatures.getDefaultValuesForFeatures().get(CONTEXT),
                dtype);
    }

    /**
     * This will return a map of <feature_name, default_value> for a given data type
     * for tfRecord type Sequence
     */
    public <T> Map<String, T> getSequenceFeaturesForDataType(DataType dtype) {
        return defaultValueExtractor(modelFeatures.getDataTypesForFeatures().get(SEQUENCE),
                modelFeatures.getDefaultValuesForFeatures().get(SEQUENCE),
                dtype);
    }

    /**
     * This will return a map of <feature_name, serving_feature_name> for a given data type
     * for tfRecord type Context
     */
    public Map<String, String> getServingNameMappingForContextFeatures() {
        return modelFeatures.getServingNameMappingForFeatures().get(CONTEXT);
    }

    /**
     * This will return a map of <feature_name, serving_feature_name> for a given data type
     * for tfRecord type Sequence
     */
    public Map<String, String> getServingNameMappingForSequenceFeatures() {
        return modelFeatures.getServingNameMappingForFeatures().get(SEQUENCE);
    }

    private <T> Map<String, T>  defaultValueExtractor(Map<String, String> featureDataTypes, Map<String, String> defaultValues, DataType dtype) {
        return featureDataTypes
                .entrySet()
                .stream()
                .filter(e -> dtype.equals(DataType.valueOf(e.getValue().toUpperCase())))
                .collect(Collectors.toMap(e -> e.getKey(), e -> (T) defaultValues.get(e.getKey())));
    }
}
