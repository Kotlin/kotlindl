/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.activation.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.junit.jupiter.api.Test

class ActivationLayersImportExportTest {
    @Test
    fun activationLayer() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                ActivationLayer(name = "test_activation_layer", activation = Activations.Exponential)
            )
        )
    }

    @Test
    fun elu() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                ELU(name = "test_elu", alpha = 0.5f)
            )
        )
    }

    @Test
    fun leakyRelu() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                LeakyReLU(name = "test_leaky_relu", alpha = 0.5f)
            )
        )
    }

    @Test
    fun prelu() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                PReLU(
                    name = "test_prelu",
                    alphaInitializer = HeNormal(),
                    alphaRegularizer = L2(),
                    sharedAxes = intArrayOf(1)
                )
            )
        )
    }

    @Test
    fun relu() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                ReLU(name = "test_relu", maxValue = 2.0f, threshold = 0.1f, negativeSlope = 2.0f)
            )
        )
    }

    @Test
    fun softmax() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                Softmax(name = "test_softmax", axis = listOf(1))
            )
        )
    }

    @Test
    fun thresholdedRelu() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100),
                ThresholdedReLU(name = "test_thresholded_relu", theta = 2f)
            )
        )
    }
}