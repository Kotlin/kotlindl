package api.keras.initializers

import api.getDType
import api.keras.shape.shapeOperand
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

internal class ParametrizedRandomNormalTest {
    val EPS = 1e-7f
    val FAN_IN = 10
    val FAN_OUT = 20
    val MEAN = 0.0f
    val STD_DEV = 3.0f
    val P1 = -4f
    val P2 = 4f

    @Test
    fun initialize() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = 2.892295f
        expected[0][1] = 2.0599146f
        expected[1][0] = -1.6870136f
        expected[1][1] = -1.1708088f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = ParametrizedRandomNormal<Float>(MEAN, STD_DEV, P1, P2, 12L)
            val operand = instance.initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), getDType(), "default_name")
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