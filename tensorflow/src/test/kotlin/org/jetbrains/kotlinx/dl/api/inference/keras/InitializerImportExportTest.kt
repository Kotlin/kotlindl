/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.junit.jupiter.api.Test

class InitializerImportExportTest {
    @Test
    fun ones() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = Ones())
            )
        )
    }

    @Test
    fun zeros() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = Zeros())
            )
        )
    }

    @Test
    fun randomNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = RandomNormal(mean = 10f, stdev = 0.1f, seed = 10))
            )
        )
    }

    @Test
    fun randomUniform() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = RandomUniform(maxVal = 10f, minVal = -10f, seed = 10))
            )
        )
    }

    @Test
    fun truncatedNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = TruncatedNormal(seed = 10))
            )
        )
    }

    @Test
    fun parametrizedTruncatedNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(
                    10, kernelInitializer = ParametrizedTruncatedNormal(
                        mean = 0.1f,
                        stdev = 2f,
                        p1 = -5f,
                        p2 = 5.1f,
                        seed = 10
                    )
                )
            )
        )
    }

    @Test
    fun glorotNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = GlorotNormal(seed = 10))
            )
        )
    }

    @Test
    fun glorotUniform() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = GlorotUniform(seed = 10))
            )
        )
    }

    @Test
    fun heNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = HeNormal(seed = 10))
            )
        )
    }

    @Test
    fun heUniform() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = HeUniform(seed = 10))
            )
        )
    }

    @Test
    fun lecunNormal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = LeCunNormal(seed = 10))
            )
        )
    }

    @Test
    fun lecunUniform() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = LeCunUniform(seed = 10))
            )
        )
    }

    @Test
    fun identity() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = Identity(gain = 2.0f))
            )
        )
    }

    @Test
    fun constant() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Dense(10, kernelInitializer = Constant(constantValue = 2.0f))
            )
        )
    }

    @Test
    fun orthogonal() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(2),
                Dense(2, kernelInitializer = Orthogonal(gain = 0.5f, seed = 10))
            )
        )
    }
}