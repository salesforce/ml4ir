from ml4ir.model.losses.loss_base import ListwiseLossBase
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers


class RankOneListNet(ListwiseLossBase):
    def _make_loss_fn(self, **kwargs):
        """
        Define a rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        """
        cce = losses.CategoricalCrossentropy(from_logits=False)
        # mask = kwargs.get("mask")

        def _loss_fn(y_true, y_pred):
            # Mask the predictions to ignore padded records
            # cce = tf.math.multiply(tf.constant(-1.),
            #                        tf.gather_nd(tf.cast(tf.math.log(y_pred), tf.float32),
            #                                     tf.equal(y_true, 1.0)))
            # cce = tf.where(tf.equal(mask, tf.constant(1.0)), cce, tf.constant(0.0))
            # cce = tf.where(tf.math.is_nan(cce), tf.constant(0.0), cce)

            # return cce(y_true, y_pred)

            return cce(y_true, y_pred)

        return _loss_fn

    def _final_activation_op(self):
        # Without masking padded records
        softmax = layers.Softmax(axis=-1, name="ranking_scores")

        def softmax_op(logits, mask):
            return softmax(logits)

        return softmax_op

        # Listwise Top 1 RankNet Loss
        def masked_softmax(logits, mask):
            """
            NOTE:
            Clip the values to a small number above 0.
            Otherwise softmax will blow up and return null values

            Eg: softmax([-200., -300., 0., 0.])
            """
            exponents = tf.clip_by_value(
                tf.math.exp(logits), tf.constant(1e-9), tf.constant(tf.float32.max)
            )

            masked_exponents = tf.math.multiply(exponents, tf.cast(mask, tf.float32))
            sum_masked_exponents = tf.expand_dims(
                tf.reduce_sum(masked_exponents, axis=-1), axis=-1
            )
            probabilities = tf.math.divide(masked_exponents, sum_masked_exponents)
            probabilities = tf.clip_by_value(
                probabilities, tf.constant(1e-9), tf.constant(1.0), name="ranking_scores"
            )
            return probabilities

        return masked_softmax
