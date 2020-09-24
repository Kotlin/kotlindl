package api.core.activation

import org.junit.jupiter.api.Test

internal class ExponentialActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            0.0f, 4.539993E-5f, 0.36787945f, 1.0f, 2.7182817f,
            22026.467f
        )

        assertActivationFunction(ExponentialActivation(), input, actual, expected)
    }
}