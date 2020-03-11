package ml4ir.inference.tensorflow;

import com.google.common.collect.Maps;
import ml4ir.inference.tensorflow.data.Document;
import ml4ir.inference.tensorflow.data.QueryContext;
import ml4ir.inference.tensorflow.utils.FeatureConfig;
import ml4ir.inference.tensorflow.utils.SequenceExampleBuilder;
import org.tensorflow.example.SequenceExample;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Helper class to build SequenceExample protobufs from base primitives and java collections
 */
public class SequenceExampleJavaBuilder {
    private final String queryString;
    private final List<Document> docs = new ArrayList<>();

    public SequenceExampleJavaBuilder(String queryString) {
        this.queryString = queryString;
    }

    public SequenceExampleJavaBuilder addDoc(String docId,
                                             Map<String, Float> floatFeatures,
                                             Map<String, Long> longFeatures,
                                             Map<String, String> stringFeatures) {
        docs.add(Document.apply(docId, floatFeatures, longFeatures, stringFeatures));
        return this;
    }

    public SequenceExampleJavaBuilder addFloatFeaturesDoc(String docId,
                                                          Map<String, Float> floatFeatures) {
        return addDoc(docId, floatFeatures, Maps.newHashMap(), Maps.newHashMap());
    }


    public SequenceExampleJavaBuilder addLongFeaturesDoc(String docId,
                                                         Map<String, Long> longFeatures) {
        return addDoc(docId, Maps.newHashMap(), longFeatures, Maps.newHashMap());
    }


    public SequenceExampleJavaBuilder addStringFeaturesDoc(String docId,
                                                          Map<String, String> stringFeatures) {
        return addDoc(docId, Maps.newHashMap(), Maps.newHashMap(), stringFeatures);
    }

    /**
     * Final builder step.
     * @return the protobuf instantiation of the query and docs-to-be-scored
     */
    public SequenceExample build() {
        return new SequenceExampleBuilder(FeatureConfig.apply())
                .apply(QueryContext.apply(queryString), docs.toArray(new Document[0]));
    }
}
