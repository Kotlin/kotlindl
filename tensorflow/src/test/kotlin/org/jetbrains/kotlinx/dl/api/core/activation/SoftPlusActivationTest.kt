/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class SoftPlusActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f)
        val expected = floatArrayOf(
            0.0f, 4.5417706E-5f, 0.31326166f, 0.6931472f, 1.3132616f,
            10.000046f
        )

        assertActivationFunction(SoftPlusActivation(), input, expected)
    }
}