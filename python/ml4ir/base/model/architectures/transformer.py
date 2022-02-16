import tensorflow as tf

"""
Implementation of transformer architecture based on tensorflow tutorial
Reference -> https://www.tensorflow.org/text/tutorials/transformer
"""


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Parameters
    -----------
    q: Tensor
        query tensor with shape == (..., seq_len_q, depth)
    k: Tensor
        key tensor with shape == (..., seq_len_k, depth)
    v: Tensor
        value tensor with shape == (..., seq_len_v, depth_v)
    mask: Tensor
        Float tensor with shape broadcastable
        to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns
    -------
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        # NOTE: Mask is used differently in our implementation
        #       because 1 - actual record and 0 - padded record
        scaled_attention_logits += ((1. - mask) * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(model_dim, feed_forward_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(feed_forward_dim, activation='relu'),  # (batch_size, seq_len, feed_forward_dim)
        tf.keras.layers.Dense(model_dim)  # (batch_size, seq_len, model_dim)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim

        assert model_dim % self.num_heads == 0

        self.depth = model_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)

        self.dense = tf.keras.layers.Dense(model_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, model_dim)
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, model_dim)

        return output, attention_weights


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(model_dim, feed_forward_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, model_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, model_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, model_dim)

        return out2
