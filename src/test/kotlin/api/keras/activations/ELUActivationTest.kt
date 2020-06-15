package api.keras.activations

import org.junit.jupiter.api.Test

internal class ELUActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            -1.0f, -0.9999546f, -0.63212055f, 0.0f, 1.0f,
            10f, 100f
        )

        assertActivationFunction(EluActivation(), input, actual, expected)
    }
}