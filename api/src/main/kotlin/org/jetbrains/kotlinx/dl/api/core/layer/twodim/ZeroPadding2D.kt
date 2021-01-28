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

/**
 * Zero-padding layer, which adds zeros on sides of image
 * @see [Tensorflow implementation](https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/layers/convolutional.py#L2765)
 */
public class ZeroPadding2D : Layer {
    public val padding: IntArray
    private val dataFormat: String
    private lateinit var inputShape: Shape

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param[padding] symmetric padding applied to width and height (same on all sides)
     * @param[dataFormat] one of [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST]
     * or [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param[name] layer name
     */
    public constructor(
        padding: Int,
        dataFormat: String?,
        name: String = ""
    ) : this(
        IntArray(4) { padding },
        dataFormat,
        name
    )

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param[padding] pair of padding values - [padding.first] represents vertical padding (applied to top and
     * bottom of image, and [padding.last] is horizontal padding (left and right sides)
     * @param[dataFormat] one of [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST]
     * or [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param[name] layer name
     */
    public constructor(
        padding: Pair<Int, Int>,
        dataFormat: String?,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.first, padding.second, padding.second),
        dataFormat,
        name
    )

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param[padding] list of padding values. Size of list must be equal to 4. Those list values maps to
     * the following paddings:
     * padding[0] -> top padding,
     * padding[1] -> bottom padding,
     * padding[2] -> left padding,
     * padding[3] -> right padding
     * @param[dataFormat] one of [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST]
     * or [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param[name] layer name
     */
    public constructor(padding: IntArray, dataFormat: String?, name: String = "") : super(name) {
        require(padding.size == 4)
        this.padding = padding
        this.dataFormat = dataFormat ?: CHANNELS_LAST
    }

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        this.inputShape = inputShape
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        require(inputShape.numDimensions() == 4) { "input tensor must have 4 dimensions" }

        return if (dataFormat == CHANNELS_FIRST) {
            Shape.make(
                inputShape.size(0),
                inputShape.size(1),
                inputShape.size(2) + padding[0] + padding[1],
                inputShape.size(3) + padding[2] + padding[3]
            )
        } else {
            Shape.make(
                inputShape.size(0),
                inputShape.size(1) + padding[0] + padding[1],
                inputShape.size(2) + padding[2] + padding[3],
                inputShape.size(3)
            )
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingOperand = tf.constant(paddingArrayToTfFormat())
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "ZeroPadding2D(padding=$padding)"
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
