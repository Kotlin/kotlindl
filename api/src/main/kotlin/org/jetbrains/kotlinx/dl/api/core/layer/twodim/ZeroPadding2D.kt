/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.twodim

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class ZeroPadding2D(
    val padding: IntArray,
    private val dataFormat: String? = CHANNELS_LAST,
    name: String = ""
) : Layer(name) {

    private lateinit var inputShape: Shape

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        this.inputShape = inputShape
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        assert(inputShape.numDimensions() == 4) { "input tensor must have 4 dimensions" }

        val normalizedPadding = normalizePaddingArray()
        return if (dataFormat == CHANNELS_FIRST) {
            Shape.make(
                inputShape.size(0),
                inputShape.size(1),
                inputShape.size(2) + normalizedPadding[0] + normalizedPadding[1],
                inputShape.size(3) + normalizedPadding[2] + normalizedPadding[3]
            )
        } else {
            Shape.make(
                inputShape.size(0),
                inputShape.size(1) + normalizedPadding[0] + normalizedPadding[1],
                inputShape.size(2) + normalizedPadding[2] + normalizedPadding[3],
                inputShape.size(3)
            )
        }
    }


    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        val paddingOperand = tf.constant(paddingArrayToTfFormat())
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    override fun getWeights(): List<Array<*>> {
        return emptyList()
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return 0
    }

    override fun toString(): String {
        return "ZeroPadding2D(padding=$padding)"
    }

    private fun normalizePaddingArray(): IntArray {
        return when (padding.size) {
            4 -> padding
            2 -> intArrayOf(padding[0], padding[0], padding[1], padding[1])
            1 -> intArrayOf(padding[0], padding[0], padding[0], padding[0])
            else -> throw IllegalArgumentException("Invalid padding argument at layer $name")
        }
    }

    private fun paddingArrayToTfFormat(): Array<IntArray> {
        val paddingFirstDim: IntArray
        val paddingSecondDim: IntArray

        when (padding.size) {
            4 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[1])
                paddingSecondDim = intArrayOf(padding[2], padding[3])
            }
            2 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[1], padding[1])
            }
            1 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[0], padding[0])
            }
            else -> throw IllegalArgumentException("Invalid padding argument at layer $name")
        }
        return paddingArraysToInputShape(paddingFirstDim, paddingSecondDim)
    }

    private fun paddingArraysToInputShape(paddingFirstDim: IntArray, paddingSecondDim: IntArray): Array<IntArray> {
        return when (inputShape.numDimensions()) {
            4 -> {
                if (dataFormat == CHANNELS_FIRST) {
                    arrayOf(intArrayOf(0, 0), intArrayOf(0, 0), paddingFirstDim, paddingSecondDim)
                } else {
                    arrayOf(intArrayOf(0, 0), paddingFirstDim, paddingSecondDim, intArrayOf(0, 0))
                }
            }
            3 -> {
                if (dataFormat == CHANNELS_FIRST) {
                    arrayOf(intArrayOf(0, 0), paddingFirstDim, paddingSecondDim)
                } else {
                    arrayOf(paddingFirstDim, paddingSecondDim, intArrayOf(0, 0))
                }
            }
            2 -> {
                arrayOf(paddingFirstDim, paddingSecondDim)
            }
            else -> throw IllegalArgumentException("Invalid input shape $inputShape")
        }
    }
}