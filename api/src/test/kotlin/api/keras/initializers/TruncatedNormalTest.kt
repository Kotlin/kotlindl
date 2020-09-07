package api.keras.initializers

import api.keras.shape.shapeOperand
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS = 1e-7f
private const val FAN_IN = 10
private const val FAN_OUT = 20

internal class TruncatedNormalTest {
    @Test
    fun initialize() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = -0.45539435f
        expected[0][1] = 0.3842017f
        expected[1][0] = 0.7207375f
        expected[1][1] = 1.2769456f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = TruncatedNormal(12L)
            val operand = instance.initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), "default_name")
            operand.asOutput().tensor().copyTo(actual)

            assertArrayEquals(
                expected[0],
                actual[0],
                EPS
            )

            assertArrayEquals(
                expected[1],
                actual[1],
                EPS
            )
        }
    }
}