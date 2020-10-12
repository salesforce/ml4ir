package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapSequenceExampleBuilder;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;


/**
 * Helper java class to encapsulate re-ranking (of e.g. search results, recommendations, ad candidates)
 * model inference.
 */
public class StringMapSequenceExampleExecutor {
    private final TFRecordExecutor executor;
    private final StringMapSequenceExampleBuilder protoBuilder;

    /**
     * @param modelBundlePath /path/to/savedmodel/tfrecord - location on local disk of the Tensorflow
     *                       {@code SavedModelBundle} (in particular the "tfrecord" subdirectory) used for inference
     * @param inputNodeName name of the Tensorflow graph node for the TFRecord-based serving signature.
     *                      This can be found by using: {@code saved_model_cli show --dir tfrecord/ --all }.  You'll see
     *                      something like the following:
     *                      <pre>
     *                      saved_model_cli show --dir tfrecord/ --all
     * MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
     * signature_def['__saved_model_init_op']:
     *   The given SavedModel SignatureDef contains the following input(s):
     *   The given SavedModel SignatureDef contains the following output(s):
     *     outputs['__saved_model_init_op'] tensor_info:
     *         dtype: DT_INVALID
     *         shape: unknown_rank
     *         name: NoOp
     *   Method name is:
     * signature_def['serving_tfrecord']:
     *   The given SavedModel SignatureDef contains the following input(s):
     *     inputs['protos'] tensor_info:
     *         dtype: DT_STRING
     *         shape: (-1)
     *         name: serving_tfrecord_protos:0
     *   The given SavedModel SignatureDef contains the following output(s):
     *     outputs['category_label'] tensor_info:
     *         dtype: DT_FLOAT
     *         shape: (-1, 1, 9)
     *         name: StatefulPartitionedCall_3:0
     *   Method name is: tensorflow/serving/predict
     *                      </pre>
     *                      In this example, {@code serving_tfrecord_protos} is the inputNodeName
     * @param outputNodeName The TF graph node to fetch the predictions, in the TFRecord-based serving signature.
     *                     In the above usage of the {@code saved_model_cli}, it is {@code StatefulPartitionedCall_3}
     * @param featureConfigPath File path on local disk to the {@code feature_config.yaml} that was used during training
     *                          of the model.
     */
    public StringMapSequenceExampleExecutor(String modelBundlePath,
                                            String inputNodeName,
                                            String outputNodeName,
                                            String featureConfigPath) {
        executor = new TFRecordExecutor(
                modelBundlePath,
                ModelExecutorConfig.apply(inputNodeName, outputNodeName)
        );
        protoBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(
                ModelFeaturesConfig.load(featureConfigPath),
                ImmutableMap.of(),
                ImmutableMap.of(),
                ImmutableMap.of()
        );
    }

    /**
     * Perform model inference
     * @param contextFeatures record-independent features (such as the search query string, the userId or cookie)
     * @param recordFeatures records to be re-ranked.  The order they are in inside this list will be the order the
     *                       predictions will appear in the {@code float[]} output
     * @return quality scores for each of the input records. No inherent range is guaranteed by the API, and decisions
     * about whether these scores should all be non-negative, less than 1.0, or even Float.NaN or not, are up to the
     * model implementer.
     */
    public float[] predict(Map<String, String> contextFeatures, List<Map<String, String>> recordFeatures) {
        SequenceExample sequenceExample = protoBuilder.build(contextFeatures, recordFeatures);
        float[] predictions = executor.apply(sequenceExample);
        return predictions;
    }
}
