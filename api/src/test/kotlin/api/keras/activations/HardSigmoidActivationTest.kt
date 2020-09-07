package api.keras.activations

import org.junit.jupiter.api.Test

internal class HardSigmoidActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            -19.5f, -1.5f, 0.3f, 0.5f, 0.7f,
            2.5f, 20.5f
        )

        assertActivationFunction(HardSigmoidActivation(), input, actual, expected)
    }
}