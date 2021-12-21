/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Zero-padding layer for 1D input (e.g. audio).
 * This layer can add zeros in the rows of the audio tensor
 * @property [padding] 2 numbers  interpreted as `(left_pad, right_pad)`.
 */
public class ZeroPadding1D : AbstractZeroPadding {
    public val padding: IntArray
    private lateinit var inputShape: Shape

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] symmetric padding applied to width(same on all sides)
     * @param [name] layer name
     */
    public constructor(
        padding: Int,
        name: String = ""
    ) : this(
        IntArray(2) { padding },
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] triple of padding values - [Pair.first] represents padding on left side
     * and [Pair.second] is padding on right side
     * @param [name] layer name
     */
    public constructor(
        padding: Pair<Int, Int>,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.second),
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] list of padding values. Size of list must be equal to 2. Those list values maps to
     * the following paddings:
     * padding[0] -> left padding,
     * padding[1] -> right padding,
     * @param [name] layer name
     */
    public constructor(padding: IntArray, name: String = "") : super(name) {
        require(padding.size == 2)
        this.padding = padding
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        this.inputShape = inputShape
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val length = inputShape.size(1) + padding[0] + padding[1]
        return Shape.make(inputShape.size(1), length, inputShape.size(2))
    }

    override fun paddingArrayToTfFormat(): Array<IntArray> {
        return arrayOf(intArrayOf(0, 0), intArrayOf(padding[0], padding[1]), intArrayOf(0, 0))
    }

    override fun toString(): String {
        return "ZeroPadding1D(padding=$padding)"
    }
}
