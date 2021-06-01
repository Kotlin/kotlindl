package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool1D
import org.junit.jupiter.api.Test

internal class GlobalAvgPooling1DTest : PoolLayerTest() {
    @Test
    fun globalAvgPool1DTest(){
        val input = Array(2, { Array(3, { FloatArray(4) { 0f } } ) } )
        val expected = Array(2, {FloatArray(4) { 0f } })
        assertGlobalAvgPool1DEquals(GlobalAvgPool1D(),input, expected )
    }
}
