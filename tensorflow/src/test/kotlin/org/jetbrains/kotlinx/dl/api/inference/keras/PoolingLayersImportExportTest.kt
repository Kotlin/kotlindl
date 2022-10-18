/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.*
import org.junit.jupiter.api.Test

class PoolingLayersImportExportTest {
    @Test
    fun avgPool1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 3),
                AvgPool1D(name = "test_avg_pool", poolSize = 4, strides = 4, padding = ConvPadding.SAME)
            )
        )
    }

    @Test
    fun avgPool2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                AvgPool2D(
                    name = "test_avg_pool",
                    poolSize = intArrayOf(1, 5, 3, 1),
                    strides = intArrayOf(1, 5, 3, 1),
                    padding = ConvPadding.SAME
                )
            )
        )
    }

    @Test
    fun avgPool3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 3),
                AvgPool3D(
                    name = "test_avg_pool",
                    poolSize = intArrayOf(1, 7, 5, 3, 1),
                    strides = intArrayOf(1, 7, 5, 3, 1),
                    padding = ConvPadding.SAME
                )
            )
        )
    }

    @Test
    fun maxPool1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 3),
                MaxPool1D(name = "test_max_pool", poolSize = 4, strides = 4, padding = ConvPadding.SAME)
            )
        )
    }

    @Test
    fun maxPool2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                MaxPool2D(
                    name = "test_max_pool",
                    poolSize = intArrayOf(1, 5, 3, 1),
                    strides = intArrayOf(1, 5, 3, 1),
                    padding = ConvPadding.SAME
                )
            )
        )
    }

    @Test
    fun maxPool3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 10),
                MaxPool3D(
                    name = "test_max_pool",
                    poolSize = intArrayOf(1, 7, 5, 3, 1),
                    strides = intArrayOf(1, 7, 5, 3, 1),
                    padding = ConvPadding.SAME
                )
            )
        )
    }

    @Test
    fun globalAvgPool1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 3),
                GlobalAvgPool1D(name = "test_global_avg_pool")
            )
        )
    }

    @Test
    fun globalAvgPool2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                GlobalAvgPool2D(name = "test_global_avg_pool")
            )
        )
    }

    @Test
    fun globalAvgPool3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 3),
                GlobalAvgPool3D(name = "test_global_avg_pool")
            )
        )
    }

    @Test
    fun globalMaxPool1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 3),
                GlobalMaxPool1D(name = "test_global_max_pool")
            )
        )
    }

    @Test
    fun globalMaxPool2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                GlobalMaxPool2D(name = "test_global_max_pool")
            )
        )
    }

    @Test
    fun globalMaxPool3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 5),
                GlobalMaxPool3D(name = "test_global_max_pool")
            )
        )
    }
}