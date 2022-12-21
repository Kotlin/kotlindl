package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class GeluActivationTest : ActivationTest() {

    @Test
    fun default() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 1.0f, 3.0f)
        val expected = floatArrayOf(-0.00404951f, -0.15865529f, 0f, 0.8413447f, 2.9959507f)
        assertActivationFunction(GeluActivation(), input, expected)
    }

    @Test
    fun approxTest() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 1.0f, 3.0f)
        val expected = floatArrayOf(-0.00363752f, -0.15880796f, 0f, 0.841192f, 2.9963627f)
        assertActivationFunction(GeluActivation(approximate = true), input, expected)
    }
}
