package api.keras.activations

import org.junit.jupiter.api.Assertions
import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

const val EPS = 1e-2f

open class ActivationTest {
    protected fun assertActivationFunction(
        instance: Activation,
        input: FloatArray,
        actual: FloatArray,
        expected: FloatArray
    ) {
        EagerSession.create().use { session ->
            val tf = Ops.create(session)

            val operand = instance.apply(tf, tf.constant(input))
            operand.asOutput().tensor().copyTo(actual)

            Assertions.assertArrayEquals(
                expected,
                actual,
                EPS
            )
        }
    }
}