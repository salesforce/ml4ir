package ml4ir.inference.tensorflow.data

import com.fasterxml.jackson.annotation.JsonProperty
import ml4ir.inference.tensorflow.utils.{InputFeatures, Label, QueryKey, Rank}

case class MF(@JsonProperty("query_key") queryKey: QueryKey,
              @JsonProperty("rank") rank: Rank,
              @JsonProperty("label") label: Label,
              @JsonProperty("features") features: List[InputFeatures])
