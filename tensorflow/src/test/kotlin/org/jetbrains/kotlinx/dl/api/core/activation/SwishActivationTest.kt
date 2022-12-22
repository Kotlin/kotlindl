/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class SwishActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val expected = floatArrayOf(
            -0.0f, -4.5388937E-4f, -0.26894143f, 0.0f, 0.7310586f,
            9.999546f, 100.0f
        )

        assertActivationFunction(SwishActivation(), input, expected)
    }
}