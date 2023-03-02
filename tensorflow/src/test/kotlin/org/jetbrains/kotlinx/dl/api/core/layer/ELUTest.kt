/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.activation.ELU
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

internal class ELUTest : ActivationLayerTest() {

    @Test
    fun defaultELU() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(-0.9502f, -0.6321f, 0.0f, 2.0f)

        assertActivationFunctionSameOutputShape(ELU(), input, expected)
    }

    @Test
    fun positiveAlphaParametrizedELU() {
        val alpha = 2.42f
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val expected = floatArrayOf(-2.2995f, -1.5297f, 0.0f, 2.0f)

        assertActivationFunctionSameOutputShape(ELU(alpha = alpha), input, expected)
    }

    @Test
    fun negativeAlphaParametrizedELU() {
        val alpha = -3.14f

        assertThrows(IllegalArgumentException::class.java) {
            assertActivationFunctionIrrelevantInputOutput(ELU(alpha = alpha))
        }
    }

    @Test
    fun zeroAlphaELU() {
        val alpha = 0.0f

        assertThrows(IllegalArgumentException::class.java) {
            assertActivationFunctionIrrelevantInputOutput(ELU(alpha = alpha))
        }
    }
}
