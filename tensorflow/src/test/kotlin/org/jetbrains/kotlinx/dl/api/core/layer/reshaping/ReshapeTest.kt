/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.LayerTest
import org.jetbrains.kotlinx.dl.api.core.layer.RunMode
import org.junit.jupiter.api.Test

internal class ReshapeTest : LayerTest() {
    @Test
    fun default() {
        val layer = Reshape(targetShape = listOf(2, 5))

        val input = arrayOf(
            FloatArray(10) { it + 10.0f },
            FloatArray(10) { it + 30.0f }
        )
        val expected = arrayOf(
            arrayOf(
                FloatArray(5) { it + 10.0f },
                FloatArray(5) { it + 15.0f }
            ),
            arrayOf(
                FloatArray(5) { it + 30.0f },
                FloatArray(5) { it + 35.0f }
            )
        )
        assertLayerOutputIsCorrect(layer, input, expected)
        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
    }

    @Test
    fun flatten() {
        val layer = Reshape(targetShape = listOf(60))

        var i = 0.0f
        val input = Array(10) { Array(2) { Array(3) { Array(2) { FloatArray(5) { i++ } } } } }

        i = 0.0f
        val expected = Array(10) { FloatArray(60) { i++ } }

        assertLayerOutputIsCorrect(layer, input, expected)
        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
    }

    @Test
    fun fromFlatToStructured() {
        val layer = Reshape(targetShape = listOf(2, 3, 2, 5))

        var i = 0.0f
        val input = Array(10) { FloatArray(60) { i++ } }

        i = 0.0f
        val expected = Array(10) { Array(2) { Array(3) { Array(2) { FloatArray(5) { i++ } } } } }

        assertLayerOutputIsCorrect(layer, input, expected)
        assertLayerOutputIsCorrect(layer, input, expected, RunMode.GRAPH)
    }
}