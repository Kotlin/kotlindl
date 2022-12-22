/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class ExponentialActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f)
        val expected = floatArrayOf(
            0.0f, 4.539993E-5f, 0.36787945f, 1.0f, 2.7182817f,
            22026.467f
        )

        assertActivationFunction(ExponentialActivation(), input, expected)
    }
}