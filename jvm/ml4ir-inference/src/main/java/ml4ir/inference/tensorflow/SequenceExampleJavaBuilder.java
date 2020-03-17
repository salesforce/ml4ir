package ml4ir.inference.tensorflow;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import ml4ir.inference.tensorflow.data.Example;
import ml4ir.inference.tensorflow.data.QueryContext;
import ml4ir.inference.tensorflow.utils.FeatureConfig;
import ml4ir.inference.tensorflow.utils.SequenceExampleBuilder;
import org.tensorflow.example.SequenceExample;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Helper class to build SequenceExample protobufs from base primitives and java collections
 */
public class SequenceExampleJavaBuilder {
    private final SequenceExampleBuilder sequenceExampleBuilder;
    private final Example context;
    private final List<Example> docs;

    public SequenceExampleJavaBuilder(FeatureConfig featureConfig,
                                      String ctxId,
                                      Map<String, Float> ctxFloats,
                                      Map<String, Long> ctxLongs,
                                      Map<String, String> ctxStrings) {
        sequenceExampleBuilder = new SequenceExampleBuilder(featureConfig);
        context = Example.apply(ctxId, ctxFloats, ctxLongs, ctxStrings);
        docs = Lists.newArrayList();
    }

    public SequenceExampleJavaBuilder(Example context, List<Example> sequence) {
        sequenceExampleBuilder = new SequenceExampleBuilder(FeatureConfig.apply());
        this.context = context;
        docs = sequence;
    }

    public static <T> SequenceExample build(Example context, List<T> sequence, Function<T, Example> exampleBuilder) {
        List<Example> examples = sequence.stream().map(exampleBuilder).collect(Collectors.toList());
        return new SequenceExampleJavaBuilder(context, examples).build();
    }

    public SequenceExampleJavaBuilder addDoc(String docId,
                                             Map<String, Float> floatFeatures,
                                             Map<String, Long> longFeatures,
                                             Map<String, String> stringFeatures) {
        docs.add(Example.apply(docId, floatFeatures, longFeatures, stringFeatures));
        return this;
    }

    /**
     * Final builder step.
     * @return the protobuf instantiation of the query and docs-to-be-scored
     */
    public SequenceExample build() {
        return sequenceExampleBuilder.apply(context, docs.toArray(new Example[0]));
    }
}
