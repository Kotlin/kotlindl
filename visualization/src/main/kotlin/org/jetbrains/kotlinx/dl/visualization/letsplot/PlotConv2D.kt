/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.GGBunch
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D

internal val FILTER_LAYERS_PERMUTATION = intArrayOf(1, 0, 2, 3)

internal val ACTIVATION_LAYERS_PERMUTATION = intArrayOf(2, 1, 0, 3)

fun filtersPlot(
    conv2DLayer: Conv2D,
    plotFill: PlotFill = PlotFill.GRAY,
    imageSize: Int = 64,
    columns: Int = 8
): GGBunch {
    @Suppress("UNCHECKED_CAST")
    val weights = conv2DLayer.weights.values.toTypedArray()[0] as TensorImageData

    val XYInOut = extractXYInputOutputAxeSizes(weights, FILTER_LAYERS_PERMUTATION)

    val plots = inputsOutputsPlots(XYInOut[2], XYInOut[3]) { i, o ->
        xyPlot(XYInOut[0], XYInOut[1], plotFill) { x, y ->
            weights[y][x][i][o]
        }
    }
    return columnPlot(plots, columns, imageSize)
}

fun modelActivationOnLayersPlot(
    model: TrainableModel,
    x: FloatArray,
    plotFill: PlotFill = PlotFill.GRAY,
    imageSize: Int = 64,
    columns: Int = 8,
): List<GGBunch> {
    val activations = model.predictAndGetActivations(x).second
    @Suppress("UNCHECKED_CAST")
    val activationArrays = activations.mapNotNull { it as? TensorImageData }

    return activationArrays.map { weights ->
        val XYInOut = extractXYInputOutputAxeSizes(weights, ACTIVATION_LAYERS_PERMUTATION)

        val plots = inputsOutputsPlots(XYInOut[2], XYInOut[3]) { i, o ->
            xyPlot(XYInOut[0], XYInOut[1], plotFill) { x, y ->
                weights[i][y][x][o]
            }
        }
        columnPlot(plots, columns, imageSize)
    }
}
