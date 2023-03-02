/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalMaxPool3D
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Test

class GlobalMaxPool3DTest : LayerTest() {
    private val input = arrayOf(
        arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(1.0f, -2.0f, 3.0f),
                    floatArrayOf(0.5f, 2.0f, 5.0f),
                    floatArrayOf(-1.0f, 3.0f, 2.0f),
                    floatArrayOf(1.5f, -1.0f, 0.5f)
                ),
                arrayOf(
                    floatArrayOf(-1.0f, 2.0f, -2.0f),
                    floatArrayOf(2.5f, 3.0f, 1.0f),
                    floatArrayOf(-2.0f, 3.0f, 2.5f),
                    floatArrayOf(-3.0f, 1.0f, 1.5f)
                ),
            ),
            arrayOf(
                arrayOf(
                    floatArrayOf(1.0f, 3.0f, 1.0f),
                    floatArrayOf(6.0f, -2.5f, 4.0f),
                    floatArrayOf(7.0f, 0.0f, 5.0f),
                    floatArrayOf(1.0f, 2.0f, 4.0f)
                ),
                arrayOf(
                    floatArrayOf(7.0f, -3.0f, 2.0f),
                    floatArrayOf(1.0f, 2.0f, 2.0f),
                    floatArrayOf(3.0f, 5.0f, -2.0f),
                    floatArrayOf(3.0f, -1.0f, 0.0f)
                ),
            ),
        ),
    )

    private val inputShape = input.shape.toLongArray()

    @Test
    fun default() {
        val layer = GlobalMaxPool3D()
        val expected = arrayOf(
            floatArrayOf(7.0f, 5.0f, 5.0f),
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        val expectedShape = longArrayOf(inputShape[0], inputShape[4])
        assertLayerComputedOutputShape(layer, expectedShape)
    }
}
