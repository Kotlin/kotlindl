package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS = 1e-6f

internal class OrthogonalTest {
    @Test
    fun `2 by 2 matrix`() {
        doTest(2, 12L, floatArrayOf(0.96555376f, 0.26020378f, 0.26020378f, -0.9655537f))
    }

    @Test
    fun `3 by 3 matrix`() {
        doTest(3)
    }

    @Test
    fun `5 by 5 matrix`() {
        doTest(5)
    }

    @Test
    fun `10 by 10 matrix`() {
        doTest(10)
    }

    private fun doTest(size: Int, seed: Long = 12L, expectedMatrix: FloatArray? = null) {
        val shape = Shape.make(size.toLong(), size.toLong())
        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Orthogonal(gain = 1.0f, seed = seed)
            val matrix = instance.initialize(
                fanIn = 10,
                fanOut = 20,
                tf = tf,
                shape = shapeOperand(tf, shape),
                name = "default_name"
            )
            val matrixTransposed = tf.linalg.transpose(matrix, tf.constant(intArrayOf(1, 0)))
            val multiplicationResult = tf.linalg.matMul(matrixTransposed, matrix)

            Assertions.assertArrayEquals(
                identityMatrix(size),
                multiplicationResult.asOutput().tensor().convertTensorToFlattenFloatArray(),
                EPS
            )

            if (expectedMatrix != null) {
                Assertions.assertArrayEquals(
                    expectedMatrix,
                    matrix.asOutput().tensor().convertTensorToFlattenFloatArray(),
                    EPS
                )
            }
        }
    }

    private fun identityMatrix(size: Int): FloatArray {
        return FloatArray(size * size) { index -> if (index % size == index / size) 1f else 0f }
    }
}
