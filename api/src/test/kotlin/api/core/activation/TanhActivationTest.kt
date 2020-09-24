package api.core.activation

import org.junit.jupiter.api.Test

internal class TanhActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            -1.0f, -1.0f, -0.7615942f, 0.0f, 0.7615942f,
            1.0f, 1.0f
        )

        assertActivationFunction(TanhActivation(), input, actual, expected)
    }
}