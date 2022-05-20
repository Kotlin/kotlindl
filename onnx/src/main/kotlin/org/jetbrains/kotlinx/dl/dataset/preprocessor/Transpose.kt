/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

/**
 * Reverse or permute the [axes] of an input tensor.
 *
 * @property [axes] Array of ints, default value is related to the typical transpose task for H, W, C to C, W, H tensor format conversion.
 */
public class Transpose(public var axes: IntArray = intArrayOf(2, 0, 1)) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val tensorShape = intArrayOf(
            inputShape.width!!.toInt(),
            inputShape.height!!.toInt(),
            inputShape.channels!!.toInt()
        )
        val ndArray = mk.ndarray<Float, D3>(data.toList(), tensorShape)
        return ndArray.transpose(*axes).toList().toFloatArray()
    }
}


/** Image DSL Preprocessing extension.*/
public fun TensorPreprocessing.transpose(sharpBlock: Transpose.() -> Unit) {
    addOperation(Transpose().apply(sharpBlock))
}
