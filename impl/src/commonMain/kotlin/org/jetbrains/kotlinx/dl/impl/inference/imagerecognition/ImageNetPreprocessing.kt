/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*

/** TF-style preprocessing. */
internal class TfStylePreprocessing : FloatArrayOperation() {
    override fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray {
        return data.map { it / 127.5f - 1 }.toFloatArray()
    }
}

/** Torch-style preprocessing. */
internal fun torchStylePreprocessing(channelsLastParameter: Boolean = true): Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    return pipeline<Pair<FloatArray, TensorShape>>()
        .rescale { scalingCoefficient = 255f }
        .normalize {
            mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            std = floatArrayOf(0.229f, 0.224f, 0.225f)
            channelsLast = channelsLastParameter
        }
}

/** Caffe-style preprocessing. */
internal fun caffeStylePreprocessing(channelsLastParameter: Boolean = true): Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    return pipeline<Pair<FloatArray, TensorShape>>()
        .normalize {
            mean = floatArrayOf(103.939f, 116.779f, 123.68f)
            std = floatArrayOf(1f, 1f, 1f)
            channelsLast = channelsLastParameter
        }
}