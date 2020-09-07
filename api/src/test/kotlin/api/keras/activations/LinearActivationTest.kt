package api.keras.activations

import org.junit.jupiter.api.Test


class LinearActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)

        assertActivationFunction(LinearActivation(), input, actual, expected)
    }
}

