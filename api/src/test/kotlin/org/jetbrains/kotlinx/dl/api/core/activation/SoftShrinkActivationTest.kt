package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test


internal class SoftShrinkActivationTest : ActivationTest() {
    @Test
    fun defaultBoundaries() {
        val input = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(-1.5f, -0.5f, 0.0f, 0.5f, 1.5f)

        assertActivationFunction(SoftShrinkActivation(), input, actual, expected)
    }

    @Test
    fun explicitBoundaries() {
        val input = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(-1.0f, 0.0f, 0.0f, 0.0f, 1.0f)

        assertActivationFunction(SoftShrinkActivation(lower = -1.0f, upper = 1.0f), input, actual, expected)
    }
}
