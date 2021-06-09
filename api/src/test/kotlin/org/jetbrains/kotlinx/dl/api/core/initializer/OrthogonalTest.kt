package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val EPS = 1e-7f
private const val FAN_IN = 10
private const val FAN_OUT = 20
private val gain: Float =  1.0f
private val seed: Long = 12L

internal class OrthogonalTest{
    @Test
    fun initialize(){
        val actual = Array(2) { FloatArray(2) { 0f } }
        val expected = Array(2) { FloatArray(2) { 0f } }
        expected[0][0] = 0.96555376f
        expected[0][1] = 0.26020378f
        expected[1][0] = 0.26020378f
        expected[1][1] = -0.9655537f

        val shape = Shape.make(2, 2)

        EagerSession.create().use { session ->
            val tf = Ops.create(session)
            val instance = Orthogonal(gain = 1.0f, seed = 12L)
            val operand = instance.initialize(fanIn = FAN_IN,fanOut = FAN_OUT,tf=tf,shape =  shapeOperand(tf, shape), name="default_name")
            operand.asOutput().tensor().copyTo(actual)

            Assertions.assertArrayEquals(
                expected[0],
                actual[0],
                EPS
            )

            Assertions.assertArrayEquals(
                expected[1],
                actual[1],
                EPS
            )

            Assertions.assertEquals(
                "Orthogonal(gain=$gain, seed=$seed)",
                instance.toString()
            )
        }
    }
}
