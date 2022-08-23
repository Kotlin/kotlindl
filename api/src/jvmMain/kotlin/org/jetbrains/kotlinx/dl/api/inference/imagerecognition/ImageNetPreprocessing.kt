/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

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
    channelsLastParameter: Boolean = true
): FloatArray {
    return pipeline<Pair<FloatArray, TensorShape>>()
        .rescale { scalingCoefficient = 255f }
        .normalize {
            mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            std = floatArrayOf(0.229f, 0.224f, 0.225f)
            channelsLast = channelsLastParameter
        }
        .apply(input to TensorShape(imageShape)).first
}

/** Caffe-style preprocessing. */
public fun caffeStylePreprocessing(
    input: FloatArray,
    imageShape: LongArray,
    channelsLastParameter: Boolean = true
): FloatArray {
    return pipeline<Pair<FloatArray, TensorShape>>()
        .normalize {
            mean = floatArrayOf(103.939f, 116.779f, 123.68f)
            std = floatArrayOf(1f, 1f, 1f)
            channelsLast = channelsLastParameter
        }
        .apply(input to TensorShape(imageShape)).first
}