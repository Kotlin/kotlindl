/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.visualisation

import jetbrains.letsPlot.GGBunch
import jetbrains.letsPlot.geom.geomTile
import jetbrains.letsPlot.gggrid
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.label.ggtitle
import jetbrains.letsPlot.scale.scaleFillGrey
import jetbrains.letsPlot.theme
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.dataset.Dataset

internal typealias TensorImageData = Array<Array<Array<FloatArray>>>

private val FEATURE_MAP_THEME = geomTile(showLegend = false) +
        scaleFillGrey() +
        theme().axisTitleBlank()
            .axisTextBlank()
            .axisTicksBlank()

fun columnPlot(plots: Iterable<Plot>, columns: Int, imageSize: Int): GGBunch =
    gggrid(plots, columns, imageSize, imageSize, fit = true)

fun featureMapPlot(xSize: Int, ySize: Int, f: (Int, Int) -> Float): Plot {

    val gridX = List(xSize) { List(ySize) { it } }.flatten()
    val gridY = List(ySize) { y -> List(xSize) { y } }.flatten()
    val gridZ = gridX.zip(gridY.reversed()).map { f(it.first, it.second) }

    return ggplot {
        x = gridX
        y = gridY
        fill = gridZ
    } + FEATURE_MAP_THEME
}

fun featureMapPlot(imageSize: Int, f: (Int, Int) -> Float): Plot =
    featureMapPlot(imageSize, imageSize, f)

fun tileImagePlot(sampleNumber: Int, dataset: Dataset, predict: (FloatArray) -> Int? = { null }): Plot {

    val imageSize = 28
    val imageData = dataset.getX(sampleNumber)
    val imageLabel = dataset.getY(sampleNumber).toInt()
    val predictedLabel = predict(imageData)
    val title = if (predictedLabel == null) {
        "Real label: $imageLabel"
    } else {
        "Real label: $imageLabel | Predicted label: $predictedLabel"
    }

    return featureMapPlot(imageSize) { x, y -> imageData[y * imageSize + x] } + ggtitle(title)
}

@Suppress("UNCHECKED_CAST")
fun filtersPlot(conv2DLayer: Conv2D, imageSize: Int = 64, columns: Int = 8): GGBunch {

    val weights = conv2DLayer.weights.values.toTypedArray()[0] as TensorImageData

    val xyio = extractXYIOSizes(weights, intArrayOf(1, 0, 2, 3))

    val plots = inputsOutputsPlots(xyio[2], xyio[3]) { i, o ->
        featureMapPlot(xyio[0], xyio[1]) { x, y ->
            weights[y][x][i][o]
        }
    }
    return columnPlot(plots, columns, imageSize)
}

@Suppress("UNCHECKED_CAST")
fun modelActivationOnLayersPlot(
    model: TrainableModel,
    x: FloatArray,
    imageSize: Int = 64,
    columns: Int = 8
): List<GGBunch> {
    val activations = model.predictAndGetActivations(x).second
    val activationArrays = activations.mapNotNull { it as? TensorImageData }

    return activationArrays.map { weights ->
        val xyio = extractXYIOSizes(weights, intArrayOf(2, 1, 0, 3))

        val plots = inputsOutputsPlots(xyio[2], xyio[3]) { i, o ->
            featureMapPlot(xyio[0], xyio[1]) { x, y ->
                weights[i][y][x][o]
            }
        }
        columnPlot(plots, columns, imageSize)
    }
}

internal fun inputsOutputsPlots(inputs: Int, outputs: Int, f: (Int, Int) -> Plot) =
    List(inputs) { i -> List(outputs) { o -> f(i, o) } }.flatten()

internal fun extractXYIOSizes(inputData: TensorImageData, permute: IntArray = intArrayOf(0, 1, 2, 3)): IntArray =
    with(IntArray(4)) {
        this[permute[0]] = inputData.size // default xSize
        this[permute[1]] = inputData[0].size // default ySize
        this[permute[2]] = inputData[0][0].size // default inputs
        this[permute[3]] = inputData[0][0][0].size // default outputs
        this
    }
