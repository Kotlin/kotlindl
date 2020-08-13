package api.keras.activations

import org.junit.jupiter.api.Test

internal class SoftPlusActivationTest : ActivationTest() {

    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            0.0f, 4.5417706E-5f, 0.31326166f, 0.6931472f, 1.3132616f,
            10.000046f
        )

        assertActivationFunction(SoftPlusActivation(), input, actual, expected)
    }
}