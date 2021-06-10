package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.junit.jupiter.api.Test

internal class GlobalAvgPooling2DTest : PoolLayerTest() {
    @Test
    fun globalAvgPool2DTest() {
        val input = Array(2, { Array(4, { Array(5, { FloatArray(3) { 0f } }) }) })
        val expected = Array(2, { FloatArray(3) { 0f } })
        assertGlobalAvgPool2DEquals(GlobalAvgPool2D(), input, expected)
    }
}
