/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

internal class Conv1DTest : ConvLayerTest() {

    @Test
    fun zeroedInputTensorWithDefaultValues() {
        val input = createFloatConv1DTensor(batchSize = 1, size = 3, channels = 1, initValue = 0.0f)
        val expected = createFloatConv1DTensor(batchSize = 1, size = 3, channels = 32, initValue = 0.0f)

        assertFloatConv1DTensorsEquals(
            Conv1D(
                name = "TestConv1D_1",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun constantInputTensorWithValidPadding() {
        val input = createFloatConv1DTensor(batchSize = 1, size = 3,channels = 1, initValue = 1.0f)
        val expected = createFloatConv1DTensor(batchSize = 1, size = 2, channels = 16, initValue = 2.0f)

        assertFloatConv1DTensorsEquals(
            Conv1D(
                name = "TestConv1D_2",
                filters = 16,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = 2,
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }
}
