/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.Figure
import jetbrains.letsPlot.geom.geomPath
import jetbrains.letsPlot.gggrid
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.label.ggtitle
import jetbrains.letsPlot.letsPlot
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.audio.wav.WavFile
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Column plot arranges the given iterable of plots in specified number of columns and
 * creates a single figure from all given plots
 *
 * @param plots that are arranged into single figure
 * @param columns specifies the number of columns in which the plots are arranged
 * @param imageSize is a height and width of the single plot in returned plots figure
 * @return a [Figure] with all given plots
 */
fun columnPlot(plots: Iterable<Plot>, columns: Int, imageSize: Int): Figure =
    gggrid(plots, columns, imageSize, imageSize, fit = true)

/**
 * Create a tile plot with weights from specified function `f(x, y)` that specifies the
 * intensity of the single tile on the plot `(x, y)` position.
 *
 * @param xSize size of X domain of `f` function as a range [0, xSize)
 * @param ySize size of Y domain of `f` function as a range [0, ySize)
 * @param plotFeature filling colors of the created plot
 * @param f function that is plotted
 * @return [Plot] for specified function on defined domain
 */
fun xyPlot(xSize: Int, ySize: Int, plotFeature: PlotFeature, f: (Int, Int) -> Float): Plot {

    val gridX = List(xSize) { List(ySize) { it } }.flatten()
    val gridY = List(ySize) { y -> List(xSize) { y } }.flatten()
    val gridZ = gridX.zip(gridY.reversed()).map { f(it.first, it.second) }

    return letsPlot {
        x = gridX
        y = gridY
        fill = gridZ
    } + FEATURE_MAP_THEME + plotFeature.scale
}

/**
 * Create a tile plot with weights from specified function `f(x, y)` that specifies the
 * intensity of the single tile on the plot `(x, y)` position.
 *
 * @param imageSize size of X and Y domains of `f` function as a range [0, imageSize)
 * @param plotFeature filling colors of the created plot
 * @param f function that is plotted
 * @return [Plot] for specified function on defined domain
 */
fun xyPlot(imageSize: Int, plotFeature: PlotFeature, f: (Int, Int) -> Float): Plot =
    xyPlot(imageSize, imageSize, plotFeature, f)

/**
 * Create a [xyPlot] for image data given as array of the following intensities of the
 * plot tiles. Function intended to use with the inputs images from some [Dataset] for
 * model as it offers to plot extra label that the specified image is labeled by (and
 * additionally supports plotting some predicted label when model prediction is given)
 *
 * @param sampleNumber index of sample in [dataset] to be plotted
 * @param dataset that contains the input data to be plotted as model input image and base label
 * @param predict function that can define the label based on model input
 * Defaults to no predict function so no model predict label is plotted.
 * @param labelEncoding mapping from output label number to some human-readable label that is plotted.
 * Defaults to identity function.
 * @param plotFeature filling colors of the created plot
 * @return [Plot] representing model sample with prediction label if available
 */
fun flattenImagePlot(
    sampleNumber: Int,
    dataset: Dataset,
    predict: (FloatArray) -> Int? = { null },
    labelEncoding: (Int) -> Any? = { it },
    plotFeature: PlotFeature = PlotFeature.GRAY
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

    return xyPlot(imageSize, plotFeature) { x, y -> imageData[y * imageSize + x] } + ggtitle(title)
}

/**
 * Create a [soundPlot] for all channels of given [WavFile]. If it is needed, the plot data can be
 * cut from the beginning or its end because there may be extra noises that disturbs visualization.
 *
 * @param wavFile to read sound data from
 * @param beginDrop part of data to drop from begin from range [0, 1]
 * @param endDrop part of data to drop from end from range [0, 1]
 * @return [Plot] representing the amplitude of sound of given [WavFile]
 */
fun soundPlot(
    wavFile: WavFile,
    beginDrop: Double = 0.0,
    endDrop: Double = 0.0
): Plot = wavFile.use { it ->
    val soundData = it.readRemainingFrames()
    val sampleRate = it.format.sampleRate
    val frames = it.frames
    val channels = it.format.numChannels
    val secondsPerFrame = 1.0 / sampleRate.toDouble()

    val dropBeginFrames = (frames * beginDrop).roundToInt()
    val dropEndFrames = (frames * endDrop).roundToInt()
    val takeFrames = max(0, frames - dropBeginFrames - dropEndFrames).toInt()
    val singleTimeData = List(takeFrames) { (it + dropBeginFrames) * secondsPerFrame }

    val channelName = List(channels) { idx -> List(takeFrames) { "channel $idx" } }.flatten()
    val xData = List(channels) { singleTimeData }.flatten()
    val yData = soundData.flatMap { it.drop(dropBeginFrames).take(takeFrames) }

    val data = mapOf(
        "channel" to channelName,
        "time [s]" to xData,
        "amplitude" to yData
    )

    letsPlot(data) {
        x = "time [s]"
        y = "amplitude"
        fill = "channel"
    } + geomPath()
}
