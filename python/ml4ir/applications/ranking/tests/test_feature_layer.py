import numpy as np

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.features import preprocessing
from ml4ir.base.features.feature_fns import categorical as categorical_fns
from ml4ir.base.features.feature_fns import sequence as sequence_fns
from ml4ir.base.config.keys import SequenceExampleTypeKey


# class RankingModelTest(RankingTestBase):

#     def test_sequence_encoding(self):
#         """
#         Unit test sequence embedding

#         """
#         batch_size = 50
#         max_length = 20
#         embedding_size = 128
#         encoding_size = 512
#         feature_info = {
#             "feature_layer_info": {
#                 "type": "numeric",
#                 "fn": "get_sequence_encoding",
#                 "args": {
#                     "embedding_size": embedding_size,
#                     "encoding_type": "bilstm",
#                     "encoding_size": encoding_size,
#                     "max_length": max_length,
#                 },
#             },
#             "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
#         }

#         """
#         Input sequence tensor should be of type integer
#         If float, it will be cast to uint8 as we use this to
#         create one-hot representation of each time step

#         If sequence tensor is a context feature, the shape can be either
#         [batch_size, max_length] or [batch_size, 1, max_length]
#         sand the method will tile the output embedding for all records.
#         """
#         sequence_tensor = np.random.randint(256, size=(batch_size, 1, max_length))

#         sequence_encoding = get_sequence_encoding(sequence_tensor, feature_info)

#         assert sequence_encoding.shape[0] == batch_size
#         assert (
#             sequence_encoding.shape[1] == 1
#             if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT
#             else self.args.max_num_records
#         )
#         assert sequence_encoding.shape[2] == encoding_size
