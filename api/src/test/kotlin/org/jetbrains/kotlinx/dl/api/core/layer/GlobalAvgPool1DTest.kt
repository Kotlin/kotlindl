/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool1D
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.junit.jupiter.api.Test

internal class GlobalAvgPool1DTest : PoolLayerTest() {
    private val input = arrayOf (
        arrayOf (
            floatArrayOf(1.0f, 2.5f, -3.0f, 2.0f),
            floatArrayOf(-3.0f, -1.0f, 5.0f, 4.5f),
            floatArrayOf(-1.0f, 0.0f, 5.0f, 4.5f),
        ),
        arrayOf(
            floatArrayOf(-1.5f, 2.0f, 1.0f, 1.0f),
            floatArrayOf(-1.0f, 7.0f, -4.0f, 1.5f),
            floatArrayOf(3.5f, 4.0f, 2.0f, -1.5f),
        )
    )

    private val inputShapeArray = longArrayOf(
        input.size.toLong(), input[0].size.toLong(), input[0][0].size.toLong()
    )

    @Test
    fun default() {
        val layer = GlobalAvgPool1D()
        val expected = arrayOf(
            floatArrayOf(-1.0f, 0.5f, 7.0f/3, 11.0f/3),
            floatArrayOf(1.0f/3, 13.0f/3, -1.0f/3, 1.0f/3)
        )
        val expectedShapeArray = longArrayOf(inputShapeArray[0], inputShapeArray[2])
        assertGlobalPoolEquals(layer, input, expected, inputShapeArray, expectedShapeArray)
        assertLayerComputedOutputShape(layer, inputShapeArray, expectedShapeArray)
    }

    @Test
    fun withChannelsFirstDataFormat() {
        val layer = GlobalAvgPool1D(dataFormat = CHANNELS_FIRST)
        val expected = arrayOf(
            floatArrayOf(2.5f/4, 5.5f/4, 8.5f/4),
            floatArrayOf(2.5f/4, 3.5f/4, 8.0f/4)
        )
        val expectedShapeArray = longArrayOf(inputShapeArray[0], inputShapeArray[1])
        assertGlobalPoolEquals(layer, input, expected, inputShapeArray, expectedShapeArray)
        assertLayerComputedOutputShape(layer, inputShapeArray, expectedShapeArray)
    }
}