package tf_api.blocks

import tf_api.TFModel
import tf_api.blocks.layers.Layer

class Sequential {
    companion object {
        fun of(vararg layers: Layer): TFModel {
            return TFModel()
        }
    }

}