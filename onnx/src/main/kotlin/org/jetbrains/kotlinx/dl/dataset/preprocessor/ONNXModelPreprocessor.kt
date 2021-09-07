/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel

/**
 * Applies the given [onnxModel] as a preprocessing stage.
 *
 * @property [onnxModel] ONNX model. It could have multiple outputs.
 * @property [outputIndex] Index of the output to be passed forward.
 */
public class ONNXModelPreprocessor(public var onnxModel: OnnxInferenceModel?, public var outputIndex: Int = 0) :
    Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val (prediction, _) = onnxModel!!.predictRawWithShapes(data)[outputIndex]
        return prediction.array()
    }
}

/** Image DSL Preprocessing extension.*/
public fun TensorPreprocessing.onnx(block: ONNXModelPreprocessor.() -> Unit) {
    customPreprocessors.add(CustomPreprocessor(ONNXModelPreprocessor(null).apply(block)))
}
