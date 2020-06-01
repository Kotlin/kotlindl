package api.keras.activations

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Operand
import org.tensorflow.op.Ops

open class ActivationTest {
    val EPS = 1e-7f
}

class LinearActivationTest : ActivationTest() {
    @Test
    fun apply() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        val actual = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance: LinearActivation<Float> = LinearActivation()
            val operand: Operand<Float> = instance.apply(tf, tf.constant(input))
            operand.asOutput().tensor().copyTo(actual)

            assertArrayEquals(
                expected,
                actual,
                EPS
            )
        }
    }
}

