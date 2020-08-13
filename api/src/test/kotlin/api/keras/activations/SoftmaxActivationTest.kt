package api.keras.activations

import org.junit.jupiter.api.Test

internal class SoftmaxActivationTest : ActivationTest() {

    @Test
    fun apply() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            5.766128E-4f, 0.001567396f, 0.0042606243f, 0.011581578f, 0.031481992f,
            0.08557693f, 0.2326222f, 0.63233274f
        )

        assertActivationFunction(SoftmaxActivation(), input, actual, expected)
    }
}