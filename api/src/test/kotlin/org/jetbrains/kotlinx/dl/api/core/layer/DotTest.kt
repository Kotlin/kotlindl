package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.merge.Dot
import org.junit.jupiter.api.Test

internal class DotTest : LayerTest() {
    @Test
    fun default() {
        val x = arrayOf(
            arrayOf(
                floatArrayOf(0f, 1f),
                floatArrayOf(2f, 3f),
                floatArrayOf(4f, 5f),
                floatArrayOf(6f, 7f),
                floatArrayOf(8f, 9f)
            )
        )
        val y = arrayOf(arrayOf(floatArrayOf(10f, 11f, 12f, 14f, 15f), floatArrayOf(15f, 16f, 17f, 18f, 19f)))
        val input = arrayOf(x, y)
        val expected = arrayOf(arrayOf(floatArrayOf(260f, 360f), floatArrayOf(320f, 445f)))
        assertLayerOutputIsCorrect(Dot(axis = intArrayOf(1, 2)), input, expected)
    }
}
