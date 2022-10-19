/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Shape
import java.util.*

internal fun Scanner.createFloatArray(shape: Shape): Any {
    return when (shape.numDimensions()) {
        0 -> nextFloat()
        1 -> create1DimFloatArray(shape)
        2 -> create2DimFloatArray(shape)
        3 -> create3DimFloatArray(shape)
        4 -> create4DimFloatArray(shape)
        else -> throw RuntimeException("The loading of tensors with 5 and more dimensions is not supported yet")
    }
}

private fun Scanner.create4DimFloatArray(shape: Shape): Array<Array<Array<FloatArray>>> {
    val result = Array(shape.size(0).toInt()) {
        Array(shape.size(1).toInt()) {
            Array(shape.size(2).toInt()) {
                FloatArray(shape.size(3).toInt()) { 0.0f }
            }
        }
    }

    for (i in result.indices) {
        for (j in result[i].indices) {
            for (k in result[i][j].indices) {
                for (m in result[i][j][k].indices) {
                    result[i][j][k][m] = nextFloat()
                }
            }
        }
    }

    return result
}

private fun Scanner.create3DimFloatArray(shape: Shape): Array<Array<FloatArray>> {
    val result = Array(shape.size(0).toInt()) {
        Array(shape.size(1).toInt()) {
            FloatArray(shape.size(2).toInt()) { 0.0f }
        }
    }

    for (i in result.indices) {
        for (j in result[i].indices) {
            for (k in result[i][j].indices) {
                result[i][j][k] = nextFloat()
            }
        }
    }

    return result
}

private fun Scanner.create2DimFloatArray(shape: Shape): Array<FloatArray> {
    val result = Array(shape.size(0).toInt()) {
        FloatArray(shape.size(1).toInt()) { 0.0f }
    }

    for (i in result.indices) {
        for (j in result[i].indices) {
            val weight = nextFloat()
            result[i][j] = weight
        }
    }

    return result
}

private fun Scanner.create1DimFloatArray(shape: Shape): FloatArray {
    val result = FloatArray(shape.size(0).toInt()) { 0.0f }

    for (i in result.indices) {
        result[i] = nextFloat()
    }

    return result
}