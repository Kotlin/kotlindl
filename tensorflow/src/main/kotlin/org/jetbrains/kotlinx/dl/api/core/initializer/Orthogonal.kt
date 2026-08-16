/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.linalg.Qr
import kotlin.math.max
import kotlin.math.min
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32

/**
 * Initializer that generates an orthogonal matrix.
 * @property [gain] Multiplicative factor to apply to the orthogonal matrix.
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [Orthogonal] initializer.
 */

public class Orthogonal(
    public val gain: Float = 1.0f,
    public val seed: Long = 12L
) : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<TInt32>,
        name: String
    ): Operand<TFloat32> {
        val dimsShape = shape.asOutput().shape().size(0)
        require(dimsShape >= 2) { "The tensor to initialize must be at least two-dimensional" }

        // Generate a random matrix
        val distOpND: Operand<TFloat32> = tf.random.statelessRandomNormal(
            shape,
            tf.constant(longArrayOf(seed, 0L)), getDType()
        )

        // Flatten the generated random matrix with the last dimension remaining
        // its original shape, so it works for conv2d
        var numRows: Long = 1
        var i = 0
        while (i < dimsShape - 1) {
            numRows *= distOpND.asOutput().shape().size(i)
            i++
        }

        val numCols = distOpND.asOutput().shape().size(i - 1)
        val flatShape = Shape.of(max(numRows, numCols), min(numRows, numCols))
        val distOp: Operand<TFloat32> = tf.reshape(distOpND, shapeOperand(tf, flatShape))

        // Compute the qr factorization
        val qrOptions = Qr.fullMatrices(false)
        val qrOp: Qr<TFloat32> = tf.linalg.qr(distOp, qrOptions)
        val qo: Operand<TFloat32> = qrOp.q()
        val ro: Operand<TFloat32> = qrOp.r()

        //Make Q uniform
        val d: Operand<TFloat32> = tf.linalg.tensorDiagPart(ro)
        var qop: Operand<TFloat32> = tf.withName(name).math.mul(qo, tf.math.sign(d))
        if (numRows < numCols) qop = tf.withName(name).linalg.transpose(qop, tf.constant(intArrayOf(1, 0)))

        return tf.math.mul(tf.reshape(qop, shape), tf.dtypes.cast(tf.constant(this.gain), getDType()))
    }

    override fun toString(): String {
        return "Orthogonal(gain=$gain, seed=$seed)"
    }
}
