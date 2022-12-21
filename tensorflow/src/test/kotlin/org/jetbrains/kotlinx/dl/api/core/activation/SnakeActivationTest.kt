package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class SnakeActivationTest : ActivationTest() {

    @Test
    fun apply() {
        val input = floatArrayOf(-1.0f, 0.0f, 1.0f)
        val expected = floatArrayOf(-0.29192656f, 0.0f, 1.7080734f)

        assertActivationFunction(SnakeActivation(), input, expected)
    }
}
