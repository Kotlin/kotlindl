package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

class HardShrinkActivationTest : ActivationTest() {
    @Test
    fun mask() {
        val input = floatArrayOf(1f, 0f, -1f)
        val expected = floatArrayOf(1f, 0f, -1f)
        assertActivationFunction(HardShrinkActivation(-0.5f, 0.5f), input, expected)
    }

    @Test
    fun setToZero() {
        val input = floatArrayOf(0.239f, 0.01f, -0.239f)
        val expected = floatArrayOf(0f, 0f, 0f)
        assertActivationFunction(HardShrinkActivation(-0.5f, 0.5f), input, expected)
    }

    @Test
    fun mixed() {
        val input = floatArrayOf(0.239f, -5f, -10f, 0.3f, -0.5f, 239f, 0.7f, -100f, -0.4f)
        val expected = floatArrayOf(0f, -5f, -10f, 0f, -0.5f, 239f, 0f, -100f, -0f)
        assertActivationFunction(HardShrinkActivation(-0.4f, 0.7f), input, expected)
    }
}
