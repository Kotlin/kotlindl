package api.keras.activations

import org.junit.jupiter.api.Test

internal class SoftSignActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            -0.990099f, -0.90909094f, -0.5f, 0.0f, 0.5f,
            0.90909094f, 0.990099f
        )

        assertActivationFunction(SoftSignActivation(), input, actual, expected)
    }
}