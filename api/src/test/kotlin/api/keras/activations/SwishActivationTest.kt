package api.keras.activations

import org.junit.jupiter.api.Test

internal class SwishActivationTest : ActivationTest() {

    @Test
    fun apply() {
        val input = floatArrayOf(-100f, -10f, -1f, 0f, 1f, 10f, 100f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(
            -0.0f, -4.5388937E-4f, -0.26894143f, 0.0f, 0.7310586f,
            9.999546f, 100.0f
        )

        assertActivationFunction(SwishActivation(), input, actual, expected)
    }
}