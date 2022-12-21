/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class HardSigmoidActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val expected = floatArrayOf(
            -19.5f, -1.5f, 0.3f, 0.5f, 0.7f,
            2.5f, 20.5f
        )

        assertActivationFunction(HardSigmoidActivation(), input, expected)
    }
}