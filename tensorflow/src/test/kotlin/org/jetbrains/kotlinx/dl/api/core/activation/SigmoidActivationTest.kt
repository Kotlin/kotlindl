/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class SigmoidActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        val expected = floatArrayOf(
            0.7310586f, 0.8807971f, 0.95257413f, 0.98201376f, 0.9933071f,
            0.99752736f, 0.999089f, 0.99966455f, 0.9998766f, 0.9999546f
        )

        assertActivationFunction(SigmoidActivation(), input, expected)
    }
}