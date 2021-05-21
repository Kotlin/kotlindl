/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.junit.jupiter.api.Test

internal class Conv2DTest : ConvLayerTest() {

    @Test
    fun zeroedInputTensorWithDefaultValues() {
        val input = createFloatConv2DTensor(batchSize = 1, rows = 3, cols = 3, allChannelsSame(1, 0.0f))
        val expected = createFloatConv2DTensor(batchSize = 1, rows = 3, cols = 3, allChannelsSame(32, 0.0f))

        assertTensorsEquals(
            Conv2D(
                name = "TestConv2D_1",
                biasInitializer = Zeros()
            ),
            input,
            expected
        )
    }

    @Test
    fun constantInputTensorWithValidPadding() {
        val input = createFloatConv2DTensor(batchSize = 1, rows = 3, cols = 3, allChannelsSame(1, 1.0f))
        val expected = createFloatConv2DTensor(batchSize = 1, rows = 2, cols = 2, allChannelsSame(16, 4.0f))

        assertTensorsEquals(
            Conv2D(
                name = "TestConv2D_2",
                filters = 16,
                kernelInitializer = Constant(1.0f),
                biasInitializer = Zeros(),
                kernelSize = longArrayOf(2, 2),
                padding = ConvPadding.VALID
            ),
            input,
            expected
        )
    }
}
