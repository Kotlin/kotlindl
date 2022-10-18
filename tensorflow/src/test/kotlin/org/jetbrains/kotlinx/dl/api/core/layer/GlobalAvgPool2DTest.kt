/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

internal class GlobalAvgPool2DTest : LayerTest() {
    private val input = arrayOf(
        arrayOf(
            arrayOf(
                floatArrayOf(1.0f, 2.5f, -3.0f, 2.0f),
                floatArrayOf(-3.0f, -1.0f, 5.0f, 4.5f),
                floatArrayOf(-1.0f, 0.0f, 5.0f, 4.5f),
            ),
            arrayOf(
                floatArrayOf(4.0f, -2.5f, -3.0f, 6.0f),
                floatArrayOf(2.0f, 1.5f, 0.0f, -2.5f),
                floatArrayOf(-3.0f, 1.0f, 0.0f, 1.5f),
            )
        ),
        arrayOf(
            arrayOf(
                floatArrayOf(-1.5f, 2.0f, 1.0f, 1.0f),
                floatArrayOf(-1.0f, 7.0f, -4.0f, 1.5f),
                floatArrayOf(3.5f, 4.0f, 2.0f, -1.5f),
            ),
            arrayOf(
                floatArrayOf(3.5f, 2.0f, 2.0f, 3.0f),
                floatArrayOf(-1.5f, 2.0f, 2.0f, -1.5f),
                floatArrayOf(-2.5f, 3.0f, 5.0f, 3.5f),
            )
        )
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun default() {
        val layer = GlobalAvgPool2D()
        val expected = arrayOf(
            floatArrayOf(0.0f, 1.5f / 6, 4.0f / 6, 16.0f / 6),
            floatArrayOf(0.5f / 6, 20.0f / 6, 8.0f / 6, 6.0f / 6)
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[3])
        assertLayerComputedOutputShape(layer, expectedShape)
    }
}
