/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.activation.ReLU
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

private const val DATA_SIZE = 4

internal class ReLUTest : ActivationLayerTest() {
    @Test
    fun defaultRelu() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 0.0f, 0.0f, 2.0f)
        val actual = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)

        assertActivationFunction(ReLU(), input, actual, expected)
    }

    @Test
    fun reluWithMaxValue() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 0.0f, 0.0f, 1.0f)
        val actual = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)

        assertActivationFunction(ReLU(maxValue = 1.0f), input, actual, expected)
    }

    @Test
    fun reluWithNegativeSlope() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val actual = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)


        val exception = Assertions.assertThrows(UnsupportedOperationException::class.java) {
            assertActivationFunction(ReLU(negativeSlope = 1.0f), input, actual, expected)
        }
        assertEquals(
            "Should be returned the LeakyRelu, but it's not supported in KotlinDL yet!",
            exception.message
        )
    }

    @Test
    fun reluWithThreshold() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 0.0f, 0.0f, 2.0f)
        val actual = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)

        assertActivationFunction(ReLU(threshold = 1.5f), input, actual, expected)
    }
}
