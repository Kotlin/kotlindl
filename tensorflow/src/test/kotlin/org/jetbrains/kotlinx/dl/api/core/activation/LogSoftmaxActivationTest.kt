/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class LogSoftmaxActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val expected = floatArrayOf(
            -200.0f, -110.0f, -101.0f, -100.0f, -99.0f,
            -90.0f, 0.0f
        )

        assertActivationFunction(LogSoftmaxActivation(), input, expected)
    }
}