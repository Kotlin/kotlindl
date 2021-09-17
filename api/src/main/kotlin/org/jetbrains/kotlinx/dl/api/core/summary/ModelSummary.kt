package org.jetbrains.kotlinx.dl.api.core.summary

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

public data class ModelSummary(
    val type: String,
    val name: String?,
    val layersSummaries: List<LayerSummary>,
    val trainableParamsCount: Long,
    val frozenParamsCount: Long
) {
    val totalParamsCount: Long = trainableParamsCount + frozenParamsCount
}

public data class LayerSummary(
    val name: String,
    val type: String,
    val outputShape: TensorShape,
    val paramsCount: Long,
    val inboundLayers: List<String>
)
