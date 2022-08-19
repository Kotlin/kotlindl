/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.util.flattenFloats

/**
 * Common preprocessing functions for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
 *
 * It takes [floatArray] as input with shape [tensorShape] and calls specific preprocessing according chosen [inputType].
 */
public fun preprocessInput(
    floatArray: FloatArray,
    tensorShape: LongArray? = null,
    inputType: InputType,
    channelsLast: Boolean = true
): FloatArray {
    return when (inputType) {
        InputType.TF -> floatArray.map { it / 127.5f - 1 }.toFloatArray()
        InputType.CAFFE -> caffeStylePreprocessing(floatArray, tensorShape!!, channelsLast)
        InputType.TORCH -> torchStylePreprocessing(floatArray, tensorShape!!, channelsLast)
    }
}

/** Torch-style preprocessing. */
public fun torchStylePreprocessing(
    input: FloatArray,
    imageShape: LongArray,
    channelsLast: Boolean = true
): FloatArray {
    require(imageShape.size == 3) { "Image shape should contain only 3 values, but contains ${imageShape.size} with values: ${imageShape.contentToString()}" }

    val height: Int
    val width: Int
    val channels: Int

    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    val scaledInput = input.map { it / 255f }.toFloatArray()
    val reshapedInput = reshapeInput(
        scaledInput,
        imageShape
    )

    if (channelsLast) {
        height = imageShape[0].toInt()
        width = imageShape[1].toInt()
        channels = imageShape[2].toInt() // TODO: should be correct for 3/4 and more dimensional

        for (i in 0 until height) {
            for (j in 0 until width) {
                for (k in 0 until channels) {
                    reshapedInput[i][j][k] = (reshapedInput[i][j][k] - mean[k]) / std[k]
                }
            }
        }

    } else {
        height = imageShape[1].toInt()
        width = imageShape[2].toInt()
        channels = imageShape[0].toInt()

        for (i in 0 until channels) {
            for (j in 0 until height) {
                for (k in 0 until width) {
                    reshapedInput[i][j][k] = (reshapedInput[i][j][k] - mean[i]) / std[i]
                }
            }
        }
    }

    return reshapedInput.flattenFloats()
}

/** Caffe-style preprocessing. */
public fun caffeStylePreprocessing(
    input: FloatArray,
    imageShape: LongArray,
    channelsLast: Boolean = true
): FloatArray {
    require(imageShape.size == 3) { "Image shape should contain only 3 values, but contains ${imageShape.size} with values: ${imageShape.contentToString()}" }

    val height: Int
    val width: Int
    val channels: Int

    val mean = floatArrayOf(103.939f, 116.779f, 123.68f)

    val reshapedInput = reshapeInput(
        input,
        imageShape
    )

    if (channelsLast) {
        height = imageShape[0].toInt()
        width = imageShape[1].toInt()
        channels = imageShape[2].toInt() // TODO: should be correct for 3/4 and more dimensional

        for (i in 0 until height) {
            for (j in 0 until width) {
                for (k in 0 until channels) {
                    reshapedInput[i][j][k] = (reshapedInput[i][j][k] - mean[k])
                }
            }
        }

    } else {
        height = imageShape[1].toInt()
        width = imageShape[2].toInt()
        channels = imageShape[0].toInt()

        for (i in 0 until channels) {
            for (j in 0 until height) {
                for (k in 0 until width) {
                    reshapedInput[i][j][k] = (reshapedInput[i][j][k] - mean[i])
                }
            }
        }
    }

    return reshapedInput.flattenFloats()
}

/** Reshapes [inputData] according [imageShape]. */
public fun reshapeInput(inputData: FloatArray, imageShape: LongArray): Array<Array<FloatArray>> {
    val dimOne = imageShape[0].toInt()
    val dimTwo = imageShape[1].toInt()
    val dimThree = imageShape[2].toInt()
    val reshaped = Array(
        dimOne
    ) { Array(dimTwo) { FloatArray(dimThree) } }

    var pos = 0
    for (i in 0 until dimOne) {
        for (j in 0 until dimTwo) {
            for (k in 0 until dimThree) {
                reshaped[i][j][k] = inputData[pos]
                pos++
            }
        }
    }

    return reshaped
}