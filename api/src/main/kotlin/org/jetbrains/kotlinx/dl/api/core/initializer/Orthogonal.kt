/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.linalg.Qr

/**
 * Initializer that generates an orthogonal matrix.
 * @property [gain] Multiplicative factor to apply to the orthogonal matrix.
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [Orthogonal] initializer.
 */

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
        val dimsShape = shape.asOutput().shape().size(0)
        require(dimsShape>=2) { "The tensor to initialize must be at least two-dimensional" }

        // Flatten the input shape with the last dimension remaining
        // its original shape so it works for conv2d
        var numRows:Long = 1
        var i:Int = 0
        while (i < dimsShape - 1) {
            numRows *= shape.asOutput().shape().size(i)
            i++
        }

        val numCols = shape.asOutput().shape().size(i-1)
        val flatShape: Shape = Shape.make(Math.max(numRows,numCols),Math.min(numRows,numCols))
        val seeds = longArrayOf(seed, 0L)

        // Generate a random matrix
        val distOp: Operand<Float> = tf.random.statelessRandomNormal(shapeOperand(tf,flatShape), tf.constant(seeds), getDType())

        // Compute the qr factorization
        val qrOptions = Qr.fullMatrices(false)
        val qrOp: Qr<Float> = tf.linalg.qr(distOp, qrOptions)
        val qo:Operand<Float> =  qrOp.q()
        val ro:Operand<Float> = qrOp.r()

        //Make Q uniform
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
