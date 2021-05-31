package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.linalg.Qr

public class Orthogonal(
    private val gain: Float =  1.0f,
    private val seed: Long = 12L
) : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        val dimsShape = shape.asOutput().shape().numDimensions()
        require(dimsShape>2) { "The numDimensions should greater than 2" }
        var numRows:Long = 1
        for (i in 0 until dimsShape-1){
            numRows*=shape.asOutput().shape().size(i)
        }
        val numCols = shape.asOutput().shape().size(dimsShape)
        val flatShape: Shape = Shape.make(Math.max(numRows,numCols),Math.min(numRows,numCols))
        val seeds = longArrayOf(seed, 0L)
        val distOp: Operand<Float> = tf.random.statelessRandomNormal(shapeOperand(tf,flatShape), tf.constant(seeds), getDType())

        val qrOptions = Qr.fullMatrices(false)
        val qrOp: Qr<Float> = tf.linalg.qr(distOp, qrOptions)
        val qo:Operand<Float> =  qrOp.q()
        val ro:Operand<Float> = qrOp.r()
        val d: Operand<Float> = tf.linalg.tensorDiagPart(ro)
        var qop: Operand<Float> = tf.withName(name).math.mul(qo, tf.math.sign(d))
        val n:Operand<Float>? = null
        if (numRows < numCols) qop = tf.withName(name).linalg.transpose(qop, n)
        return tf.math.mul(tf.reshape(qop,shape), tf.dtypes.cast(tf.constant(this.gain), getDType()));
    }

    override fun toString(): String {
        return "Orthogonal(gain=$gain, seed=$seed)"
    }
}
