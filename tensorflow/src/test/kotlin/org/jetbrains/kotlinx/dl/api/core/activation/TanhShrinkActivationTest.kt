package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class TanhShrinkActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val expected = floatArrayOf(-99f, -9f, -0.23840582f, 0f, 0.23840582f, 9f, 99f)

        assertActivationFunction(TanhShrinkActivation(), input, expected)
    }
}
