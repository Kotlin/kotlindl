package api.keras.initializers

import api.keras.shape.shapeOperand
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

internal class HeNormalTest {
    private val EPS = 1e-5f
    private val FAN_IN = 2
    private val FAN_OUT = 4
    private val SEED = 12L
    private val DEFAULT_LAYER_NAME = "default_name"

    @Test
    fun initialize() {
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = 0.4739421f
        expected[0][1] = -0.15234587f
        expected[1][0] = 0.12772104f
        expected[1][1] = -0.7704968f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = HeNormal(seed = SEED)
            val operand =
                instance.initialize(FAN_IN, FAN_OUT, tf, shapeOperand(tf, shape), DEFAULT_LAYER_NAME)
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