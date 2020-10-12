package ml4ir.inference.tensorflow;

import com.google.common.collect.ImmutableMap;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.StringMapExampleBuilder;
import org.tensorflow.example.Example;
import java.util.Map;

/**
 * Helper java class to encapsulate multi-class classification and regression model inference.
 */
public class StringMapExampleExecutor {
    private final TFRecordExecutor executor;
    private final StringMapExampleBuilder protoBuilder;

    /**
     *
     * @param modelBundlePath location on local disk of the Tensorflow {@code SavedModelBundle} (in particular the
     *                        "tfrecord" subdirectory) used for inference)
     * @param inputNodeName name of the Tensorflow graph node for the TFRecord-based serving signature.  This can be
     *                      found by using: {@code saved_model_cli show --dir /path/to/savedmodel/tfrecord --all }.
     *                      You'll see something like the following:
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
    public StringMapExampleExecutor(String modelBundlePath,
                                    String inputNodeName,
                                    String outputNodeName,
                                    String featureConfigPath) {
        executor = new TFRecordExecutor(
                modelBundlePath,
                ModelExecutorConfig.apply(inputNodeName, outputNodeName)
        );
        protoBuilder = StringMapExampleBuilder.withFeatureProcessors(
                ModelFeaturesConfig.load(featureConfigPath),
                ImmutableMap.of(),
                ImmutableMap.of(),
                ImmutableMap.of()
        );
    }

    /**
     * Perform model inference
     * @param features keys are the feature names, and must correspond to the {@code serving_info.name} value of an
     *                 entry in the {@code feature_config.yaml}. NOTE: this is <b>not</b> the value of the "name" field
     *                 of the feature or the "node_name" field. Also note: fields used for
     *                 {@code org.tensorflow.example.Example} inference must all have {@code tfrecord_type} set as
     *                 "context" (as Examples are all "context" and no "sequence" - while {@code SequenceExample} has
     *                 both
     * @return floating-point predictions on the input. Up to model implementations to dictate whether these are
     * probabilities which sum to one, and how many there will be.
     */
    public float[] predict(Map<String, String> features) {
        Example example = protoBuilder.apply(features);
        float[] predictions = executor.apply(example);
        return predictions;
    }
}
