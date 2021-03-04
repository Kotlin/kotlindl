/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.junit.Ignore
import org.junit.jupiter.api.Test

internal class Conv2DTest : ConvLayerTest() {
    @Ignore
    @Test
    fun conv2d() {
        val input = Array(1) {
            Array(2) {
                Array(2) {
                    FloatArray(1)
                }
            }
        }

        for (i in 0..1)
            for (j in 0..1)
                input[0][i][j][0] = 1.0f


        val expected = Array(1) {
            Array(2) {
                Array(2) {
                    FloatArray(1)
                }
            }
        }

        for (i in 0..1)
            for (j in 0..1)
                expected[0][i][j][0] = 1.0f

        val actual = Array(1) {
            Array(2) {
                Array(2) {
                    FloatArray(1)
                }
            }
        }

        for (i in 0..1)
            for (j in 0..1)
                actual[0][i][j][0] = 1.0f


        assertActivationFunction(
            Conv2D(name = "TestConv2D_1", filters = 1, kernelInitializer = HeNormal(12L), biasInitializer = Zeros()),
            input,
            actual,
            expected
        )
    }
}
