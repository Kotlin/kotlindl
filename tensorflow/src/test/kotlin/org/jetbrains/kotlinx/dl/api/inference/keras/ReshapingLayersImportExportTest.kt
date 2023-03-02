/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.*
import org.junit.jupiter.api.Test

class ReshapingLayersImportExportTest {
    @Test
    fun cropping1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 2),
                Cropping1D(name = "test_cropping", cropping = intArrayOf(1, 2))
            )
        )
    }

    @Test
    fun cropping2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                Cropping2D(name = "test_cropping", cropping = arrayOf(intArrayOf(1, 2), intArrayOf(3, 4)))
            )
        )
    }

    @Test
    fun cropping3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 2),
                Cropping3D(
                    name = "test_cropping",
                    cropping = arrayOf(intArrayOf(1, 2), intArrayOf(3, 4), intArrayOf(5, 6))
                )
            )
        )
    }

    @Test
    fun upSampling1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 2),
                UpSampling1D(name = "test_upsampling", size = 4)
            )
        )
    }

    @Test
    fun upSampling2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                UpSampling2D(name = "test_upsampling", size = intArrayOf(1, 4))
            )
        )
    }

    @Test
    fun upSampling3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 2),
                UpSampling3D(name = "test_upsampling", size = intArrayOf(1, 4, 5))
            )
        )
    }

    @Test
    fun zeroPadding1D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 2),
                ZeroPadding1D(name = "test_zero_padding", padding = intArrayOf(1, 2))
            )
        )
    }

    @Test
    fun zeroPadding2D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 3),
                ZeroPadding2D(name = "test_zero_padding", padding = intArrayOf(1, 2, 3, 4), dataFormat = CHANNELS_LAST)
            )
        )
    }

    @Test
    fun zeroPadding3D() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100, 100, 2),
                ZeroPadding3D(name = "test_zero_padding", padding = intArrayOf(1, 2, 3, 4, 5, 6))
            )
        )
    }

    @Test
    fun flatten() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100),
                Flatten(name = "test_flatten")
            )
        )
    }

    @Test
    fun permute() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(100, 100),
                Permute(name = "test_permute", dims = intArrayOf(2, 1))
            )
        )
    }

    @Test
    fun repeatVector() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                RepeatVector(name = "test_repeat_vector", n = 2)
            )
        )
    }

    @Test
    fun reshape() {
        LayerImportExportTest.run(
            Sequential.of(
                Input(10),
                Reshape(name = "test_reshape", targetShape = listOf(5, 5))
            )
        )
    }
}