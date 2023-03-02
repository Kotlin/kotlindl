/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape.Companion.tail
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.getFloatArrayWithShape

/**
 * Applies the given [onnxModel] as a preprocessing stage.
 *
 * @property [onnxModel] ONNX model. It could have multiple outputs.
 * @property [outputIndex] Index of the output to be passed forward.
 */
public class ONNXModelPreprocessor(public var onnxModel: OnnxInferenceModel?, public var outputIndex: Int = 0) :
    Operation<FloatData, FloatData> {
    override fun apply(input: FloatData): FloatData {
        val (prediction, rawShape) = onnxModel!!.predict(input) { output ->
            return@predict output.getFloatArrayWithShape(outputIndex)
        }
        return prediction to TensorShape(rawShape.tail())
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return TensorShape(onnxModel!!.outputShape)
    }
}

/** Image DSL Preprocessing extension.*/
public fun <I> Operation<I, FloatData>.onnx(block: ONNXModelPreprocessor.() -> Unit): Operation<I, FloatData> {
    return PreprocessingPipeline(this, ONNXModelPreprocessor(null).apply(block))
}
