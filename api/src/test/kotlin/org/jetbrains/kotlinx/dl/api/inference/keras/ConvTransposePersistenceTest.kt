/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv3DTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.junit.jupiter.api.Test

class ConvTransposePersistenceTest {
    @Test
    fun conv1DTranspose() {
        LayerPersistenceTest.run(
            Sequential.of(
                Input(dims = longArrayOf(3)),
                Conv1DTranspose(
                    filters = 5,
                    kernelLength = 5,
                    strides = intArrayOf(1, 2, 1),
                    activation = Activations.Tanh,
                    dilations = intArrayOf(1, 2, 1),
                    kernelInitializer = HeUniform(),
                    biasInitializer = HeUniform(),
                    kernelRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2(),
                    padding = ConvPadding.VALID,
                    outputPadding = intArrayOf(0, 0, 1, 1, 0, 0)
                )
            )
        )
    }

    @Test
    fun conv2DTranspose() {
        LayerPersistenceTest.run(
            Sequential.of(
                Input(dims = longArrayOf(3, 3)),
                Conv2DTranspose(
                    filters = 5,
                    kernelSize = intArrayOf(5, 5),
                    strides = intArrayOf(1, 2, 4, 1),
                    activation = Activations.Tanh,
                    dilations = intArrayOf(1, 2, 4, 1),
                    kernelInitializer = HeUniform(),
                    biasInitializer = HeUniform(),
                    kernelRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2(),
                    padding = ConvPadding.VALID,
                    outputPadding = intArrayOf(0, 0, 1, 1, 2, 2, 0, 0)
                )
            )
        )
    }

    @Test
    fun conv3DTranspose() {
        LayerPersistenceTest.run(
            Sequential.of(
                Input(dims = longArrayOf(3, 3, 3)),
                Conv3DTranspose(
                    filters = 5,
                    kernelSize = intArrayOf(5, 5, 5),
                    strides = intArrayOf(1, 2, 4, 2, 1),
                    activation = Activations.Tanh,
                    dilations = intArrayOf(1, 2, 4, 2, 1),
                    kernelInitializer = HeUniform(),
                    biasInitializer = HeUniform(),
                    kernelRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2(),
                    padding = ConvPadding.VALID,
                )
            )
        )
    }
}