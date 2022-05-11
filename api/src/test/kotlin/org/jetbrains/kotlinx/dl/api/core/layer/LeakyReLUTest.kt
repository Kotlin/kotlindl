/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.activation.LeakyReLU
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

internal class LeakyReLUTest : ActivationLayerTest() {
    @Test
    fun default() {
        val input = floatArrayOf(-1.0f, -2.0f, 0.0f, 3.5f, 0.5f)
        val expected = floatArrayOf(-0.3f, -0.6f, 0.0f, 3.5f, 0.5f)

        assertActivationFunctionSameOutputShape(LeakyReLU(), input, expected)
    }

    @Test
    fun withNegativeSlope() {
        val input = floatArrayOf(-1.0f, -2.0f, 0.0f, 3.5f, 1.0f)
        val expected = floatArrayOf(-2.0f, -4.0f, 0.0f, 3.5f, 1.0f)

        assertActivationFunctionSameOutputShape(
            LeakyReLU(alpha = 2.0f), input, expected
        )
    }

    @Test
    fun withInvalidNegativeSlope() {
        val exception = Assertions.assertThrows(IllegalArgumentException::class.java) {
            assertActivationFunctionIrrelevantInputOutput(LeakyReLU(alpha = -1.0f))
        }

        assertEquals(
            "Alpha -1.0 should be >= 0.0.",
            exception.message
        )
    }
}
