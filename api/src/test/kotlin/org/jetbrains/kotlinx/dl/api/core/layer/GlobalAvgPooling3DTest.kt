package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool3D
import org.junit.jupiter.api.Test

internal class GlobalAvgPooling3DTest : PoolLayerTest() {
    @Test
    fun globalAvgPool3DTest(){
        val input = Array(2,{ Array(3,{ Array(4, { Array(5, { FloatArray(6) { 0f } }) }) }) })
        val expected = Array(2, {FloatArray(6) { 0f } })
        assertGlobalAvgPool3DEquals(GlobalAvgPool3D(),input, expected )
    }
}
