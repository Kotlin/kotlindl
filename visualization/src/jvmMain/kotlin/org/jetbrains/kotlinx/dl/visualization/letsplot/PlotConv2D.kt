/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.Figure
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.weights

internal val FILTER_LAYERS_PERMUTATION = intArrayOf(1, 0, 2, 3)

internal val ACTIVATION_LAYERS_PERMUTATION = intArrayOf(2, 1, 0, 3)

/**
 * Create a column plot of tile plots for weights of Conv2D layer filters.
 *
 * @param conv2DLayer which weights will be changed to tile plot
 * @param plotFeature filling colors of the created plot
 * @param imageSize size of width and height of single plot in px
 * @param columns number of columns in which the single filters plots are arranged
 * @return a figure representing the weights plots
 */
fun filtersPlot(
    conv2DLayer: Conv2D,
    plotFeature: PlotFeature = PlotFeature.GRAY,
    imageSize: Int = 64,
    columns: Int = 8
): Figure {
    @Suppress("UNCHECKED_CAST")
    val weights = conv2DLayer.weights.values.toTypedArray()[0] as TensorImageData

    val xyInOut = extractXYInputOutputAxeSizes(weights, FILTER_LAYERS_PERMUTATION)

    val plots = cartesianProductIndices(xyInOut[2], xyInOut[3]).map { (i, o) ->
        xyPlot(xyInOut[0], xyInOut[1], plotFeature) { x, y ->
            weights[y][x][i][o]
        }
    }
    return columnPlot(plots, columns, imageSize)
}

/**
 * Create a list of columns plots for model activation on layers.
 * The model is evaluated on given input and the obtained activations arrays
 * of the following layers are converted into separated figures with columns
 * plots of the weights for the filters in [Conv2D] layers
 *
 * @param model that is evaluated to get the activations on its weights
 * @param x input for model evaluation
 * @param plotFeature filling colors of the created plot
 * @param imageSize size of width and height of single plot in px
 * @param columns number of columns in which the single filters plots are arranged
 * @return list of figures representing the activations plots for model evaluation
 */
fun modelActivationOnLayersPlot(
    model: TrainableModel,
    x: FloatArray,
    plotFeature: PlotFeature = PlotFeature.GRAY,
    imageSize: Int = 64,
    columns: Int = 8,
): List<Figure> {
    val activations = model.predictAndGetActivations(x).second

    @Suppress("UNCHECKED_CAST")
    val activationArrays = activations.mapNotNull { it as? TensorImageData }

    return activationArrays.map { weights ->
        val xyInOut = extractXYInputOutputAxeSizes(weights, ACTIVATION_LAYERS_PERMUTATION)

        val plots = cartesianProductIndices(xyInOut[2], xyInOut[3]).map { (i, o) ->
            xyPlot(xyInOut[0], xyInOut[1], plotFeature) { x, y ->
                weights[i][y][x][o]
            }
        }
        columnPlot(plots, columns, imageSize)
    }
}
