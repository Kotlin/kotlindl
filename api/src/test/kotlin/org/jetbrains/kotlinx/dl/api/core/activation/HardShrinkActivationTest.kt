package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

class HardShrinkActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(1f, 0f, 1f)
        val expected = floatArrayOf(1f, 0f, 1f)
        val actual = floatArrayOf(0f, 0f, 0f)
        assertActivationFunction(HardShrinkActivation(-0.5f, 0.5f), input, actual, expected)
    }
}
