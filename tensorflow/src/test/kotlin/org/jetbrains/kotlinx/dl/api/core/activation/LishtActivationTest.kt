/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class LishtActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(Float.NEGATIVE_INFINITY, -5f, -0.5f, 1f, 1.2f, 2f, 3f, Float.POSITIVE_INFINITY)
        val expected = floatArrayOf(
            Float.POSITIVE_INFINITY,
            4.999546f,
            0.23105858f,
            0.7615942f,
            1.0003856f,
            1.9280552f,
            2.9851642f,
            Float.POSITIVE_INFINITY
        )

        assertActivationFunction(LishtActivation(), input, expected)
    }
}