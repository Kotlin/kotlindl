/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.extension

import org.tensorflow.Tensor
import java.nio.FloatBuffer

/** Copies tensor data to float array. */
public fun Tensor<*>.convertTensorToFlattenFloatArray(): FloatArray {
    val buffer = FloatBuffer.allocate(numElements())
    writeTo(buffer)
    return buffer.array()
}

/** Copies tensor to multidimensional float array. Array rank is equal to tensor rank. */
public fun Tensor<*>.convertTensorToMultiDimArray(): Array<*> {
    val tensorForCopying = this

    val shape = tensorForCopying.shape()
    val dst: Array<*>

    return when (shape.size) {

        1 -> {
            val oneDimDst = FloatArray(shape[0].toInt()) { 0.0f }
            tensorForCopying.copyTo(oneDimDst)
            dst = Array(oneDimDst.size) { 0.0f }
            for (i in oneDimDst.indices) dst[i] = oneDimDst[i]
            dst
        }
        2 -> {
            dst = Array(shape[0].toInt()) { FloatArray(shape[1].toInt()) }

            tensorForCopying.copyTo(dst)
            dst
        }
        3 -> {
            dst = Array(shape[0].toInt()) {
                Array(shape[1].toInt()) {
                    FloatArray(shape[2].toInt())
                }
            }
            tensorForCopying.copyTo(dst)
            dst
        }
        4 -> {
            dst = Array(shape[0].toInt()) {
                Array(shape[1].toInt()) {
                    Array(shape[2].toInt()) {
                        FloatArray(shape[3].toInt())
                    }
                }
            }
            tensorForCopying.copyTo(dst)
            dst
        }
        else -> {
            throw UnsupportedOperationException("Parsing for ${shape.size} dimensions is not supported yet!")
        }
    }
}