package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.tensorflow.op.core.Variable

public interface TrainableLayer : ParametrizedLayer {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could not be changed during the training.
     */
    public var isTrainable: Boolean
}

/**
 * Returns amount of parameters
 */
public val Layer.isTrainable: Boolean
    get() = if (this is TrainableLayer) isTrainable else false

/**
 * Returns a list of non-trainable, 'frozen' variables used in layers.
 */
public fun TensorFlowInferenceModel.frozenLayerVariables(): List<VariableDto> = layers
    .filterIsInstance<ParametrizedLayer>()
    .filter { it !is TrainableLayer || !it.isTrainable }
    .flatMap { it.variables }

/**
 * Returns a list of trainable variables used in layers.
 */
public fun TensorFlowInferenceModel.trainableLayerVariables(): List<VariableDto> = layers
    .filterIsInstance<TrainableLayer>()
    .filter { it.isTrainable }
    .flatMap { it.variables }