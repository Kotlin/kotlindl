/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.junit.jupiter.api.Test

class ConvolutionalLayersImportExportTest {
    @Test
    fun conv1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(dims = longArrayOf(256, 1)),
                Conv1D(
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
                    useBias = false,
                    name = "test_conv1D"
                )
            )
        )
    }

    @Test
    fun conv2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(dims = longArrayOf(256, 256, 3)),
                Conv2D(
                    filters = 16,
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
                    useBias = false,
                    name = "test_conv2D"
                )
            )
        )
    }

    @Test
    fun conv3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(dims = longArrayOf(10, 256, 256, 3)),
                Conv3D(
                    filters = 16,
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
                    useBias = false,
                    name = "test_conv3D"
                )
            )
        )
    }

    @Test
    fun conv1DTranspose() {
        LayerImportExportTest.run(
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
                    outputPadding = intArrayOf(0, 0, 1, 1, 0, 0),
                    useBias = false,
                    name = "test_conv1DTranspose"
                )
            )
        )
    }

    @Test
    fun conv2DTranspose() {
        LayerImportExportTest.run(
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
                    outputPadding = intArrayOf(0, 0, 1, 1, 2, 2, 0, 0),
                    useBias = false,
                    name = "test_conv2DTranspose"
                )
            )
        )
    }

    @Test
    fun conv3DTranspose() {
        LayerImportExportTest.run(
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
                    useBias = false,
                    name = "test_conv3DTranspose"
                )
            )
        )
    }

    @Test
    fun separableConv() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(dims = longArrayOf(30, 30, 3)),
                SeparableConv2D(
                    filters = 10,
                    kernelSize = intArrayOf(5, 5),
                    strides = intArrayOf(1, 2, 2, 1),
                    dilations = intArrayOf(1, 2, 2, 1),
                    activation = Activations.Tanh,
                    depthMultiplier = 2,
                    depthwiseInitializer = HeUniform(),
                    pointwiseInitializer = HeUniform(),
                    biasInitializer = HeUniform(),
                    depthwiseRegularizer = L2(),
                    pointwiseRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2(),
                    padding = ConvPadding.VALID,
                    useBias = false,
                    name = "test_separable_conv2D"
                )
            )
        )
    }

    @Test
    fun depthwiseConv() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(dims = longArrayOf(30, 30, 3)),
                DepthwiseConv2D(
                    kernelSize = intArrayOf(5, 5),
                    strides = intArrayOf(1, 2, 2, 1),
                    dilations = intArrayOf(1, 2, 2, 1),
                    activation = Activations.Tanh,
                    depthMultiplier = 2,
                    depthwiseInitializer = HeUniform(),
                    biasInitializer = HeUniform(),
                    depthwiseRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2(),
                    padding = ConvPadding.VALID,
                    useBias = false,
                    name = "test_depthwise_conv"
                )
            )
        )
    }
}