/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.junit.jupiter.api.Test

class DropoutBatchNormImportExportTest {
    @Test
    fun dropout() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dropout(name = "test_dropout", rate = 0.2f)
            )
        )
    }

    @Test
    fun batchNorm() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                BatchNorm(
                    name = "test_batch_norm", axis = listOf(1),
                    momentum = 0.9, center = false, epsilon = 0.002, scale = false,
                    gammaInitializer = HeUniform(), betaInitializer = HeNormal(),
                    betaRegularizer = L2(), gammaRegularizer = L2(),
                    movingMeanInitializer = HeNormal(), movingVarianceInitializer = HeUniform()
                )
            )
        )
    }
}