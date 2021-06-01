/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.GGBunch
import jetbrains.letsPlot.gggrid
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.label.ggtitle
import org.jetbrains.kotlinx.dl.dataset.Dataset

fun columnPlot(plots: Iterable<Plot>, columns: Int, imageSize: Int): GGBunch =
    gggrid(plots, columns, imageSize, imageSize, fit = true)

fun xyPlot(xSize: Int, ySize: Int, plotFill: PlotFill, f: (Int, Int) -> Float): Plot {

    val gridX = List(xSize) { List(ySize) { it } }.flatten()
    val gridY = List(ySize) { y -> List(xSize) { y } }.flatten()
    val gridZ = gridX.zip(gridY.reversed()).map { f(it.first, it.second) }

    return ggplot {
        x = gridX
        y = gridY
        fill = gridZ
    } + FEATURE_MAP_THEME + plotFill.scale
}

fun xyPlot(imageSize: Int, plotFill: PlotFill, f: (Int, Int) -> Float): Plot =
    xyPlot(imageSize, imageSize, plotFill, f)

fun flattenImagePlot(
    sampleNumber: Int,
    dataset: Dataset,
    predict: (FloatArray) -> Int? = { null },
    plotFill: PlotFill = PlotFill.GRAY,
    labelEncoding: (Int) -> Any? = { it }
): Plot {
    val imageSize = 28
    val imageData = dataset.getX(sampleNumber)
    val imageLabel = dataset.getY(sampleNumber).toInt().run(labelEncoding)
    val predictedLabel = predict(imageData)?.run(labelEncoding)
    val title = if (predictedLabel == null) {
        "Real label: $imageLabel"
    } else {
        "Real label: $imageLabel | Predicted label: $predictedLabel"
    }

    return xyPlot(imageSize, plotFill) { x, y -> imageData[y * imageSize + x] } + ggtitle(title)
}
