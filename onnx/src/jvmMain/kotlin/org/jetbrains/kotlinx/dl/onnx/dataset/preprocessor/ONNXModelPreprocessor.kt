/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.dataset.preprocessor

import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArrayWithShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * Applies the given [onnxModel] as a preprocessing stage.
 *
 * @property [onnxModel] ONNX model. It could have multiple outputs.
 * @property [outputIndex] Index of the output to be passed forward.
 */
public class ONNXModelPreprocessor(public var onnxModel: OnnxInferenceModel?, public var outputIndex: Int = 0) :
    Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    override fun apply(input: Pair<FloatArray, TensorShape>): Pair<FloatArray, TensorShape> {
        val (prediction, rawShape) = onnxModel!!.predictRaw(input.first) { output ->
            return@predictRaw output.getFloatArrayWithShape(outputIndex)
        }
        return prediction to TensorShape(rawShape)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return TensorShape(onnxModel!!.outputShape)
    }
}

/** Image DSL Preprocessing extension.*/
public fun <I> Operation<I, Pair<FloatArray, TensorShape>>.onnx(block: ONNXModelPreprocessor.() -> Unit): Operation<I, Pair<FloatArray, TensorShape>> {
    return PreprocessingPipeline(this, ONNXModelPreprocessor(null).apply(block))
}
