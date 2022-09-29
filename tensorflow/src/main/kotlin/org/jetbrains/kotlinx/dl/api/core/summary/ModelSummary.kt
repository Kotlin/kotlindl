package org.jetbrains.kotlinx.dl.api.core.summary

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * The common information about model.
 */
public data class ModelSummary(
    /** The model type. */
    val type: String,
    /** The model name. */
    val name: String?,
    /** The summary of the all layers included in the model. */
    val layersSummaries: List<LayerSummary>,
    /** The number of trainable parameters. */
    val trainableParamsCount: Long,
    /** The number of frozen parameters. */
    val frozenParamsCount: Long
) {
    /** The total number of model's parameters. */
    val totalParamsCount: Long = trainableParamsCount + frozenParamsCount
}

/**
 * The common information about layer.
 */
public data class LayerSummary(
    /** The layer name. */
    val name: String,
    /** The layer type. */
    val type: String,
    /** The output shape of the layer. */
    val outputShape: TensorShape,
    /** The total number of layer's parameters. */
    val paramsCount: Long,
    /** Input layers for the described layer. */
    val inboundLayers: List<String>
)
