/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2L1
import org.junit.jupiter.api.Test

class CoreLayersImportExportTest {
    @Test
    fun inputLayerSequential() {
        LayerImportExportTest.run(Sequential.of(Input(4)))
        LayerImportExportTest.run(Sequential.of(Input(128, 128)))
        LayerImportExportTest.run(Sequential.of(Input(128, 128, 3)))
        LayerImportExportTest.run(Sequential.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun inputLayerFunctional() {
        LayerImportExportTest.run(Functional.of(Input(10)))
        LayerImportExportTest.run(Functional.of(Input(128, 128)))
        LayerImportExportTest.run(Functional.of(Input(128, 128, 3)))
        LayerImportExportTest.run(Functional.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun denseLayer() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(
                    name = "test_dense",
                    outputSize = 10,
                    activation = Activations.Tanh,
                    kernelInitializer = HeNormal(),
                    biasInitializer = HeUniform(),
                    kernelRegularizer = L2(),
                    biasRegularizer = L2(),
                    activityRegularizer = L2L1(),
                    useBias = true
                )
            )
        )
    }
}
