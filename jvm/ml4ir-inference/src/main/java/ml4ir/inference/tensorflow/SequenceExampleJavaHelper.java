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

public class SequenceExampleJavaHelper {
    private String queryString;
    private List<Document> docs = new ArrayList<>();

    public SequenceExampleJavaHelper(String queryString) {
        this.queryString = queryString;
    }

    public SequenceExampleJavaHelper addDoc(String docId,
                                            Map<String, Float> floatFeatures) {
        return addDoc(docId, floatFeatures, Maps.newHashMap(), Maps.newHashMap());
    }

    public SequenceExampleJavaHelper addDoc(String docId,
                                            Map<String, Float> floatFeatures,
                                            Map<String, Long> longFeatures,
                                            Map<String, String> stringFeatures) {
        docs.add(Document.apply(docId, floatFeatures, longFeatures, stringFeatures));
        return this;
    }

    public SequenceExample build() {
        return new SequenceExampleBuilder(FeatureConfig.apply())
                .apply(new QueryContext(queryString, "", null),
                       docs.toArray(new Document[0]));
    }
}
