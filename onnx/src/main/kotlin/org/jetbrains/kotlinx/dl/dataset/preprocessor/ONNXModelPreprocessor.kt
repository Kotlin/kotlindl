/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.reshape4DTo1D
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel

/**
 * Applies the final image preprocessing that is specific for each of available models trained on ImageNet according chosen [modelType].
 *
 * @property [modelType] One the supported models pre-trained on ImageNet.
 */
public class ONNXModelPreprocessor(public var onnxModel: OnnxInferenceModel?) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        //val tensorShape = longArrayOf(inputShape.width!!, inputShape.height!!, inputShape.channels)

        val prediction = onnxModel!!.predictRaw(data)
        return reshape4DTo1D(predictRaw as Array<Array<Array<FloatArray>>>, 100352)
    }
}

/** Image DSL Preprocessing extension.*/
public fun TensorPreprocessing.onnx(block: ONNXModelPreprocessor.() -> Unit) {
    customPreprocessor = CustomPreprocessor(ONNXModelPreprocessor(null).apply(block))
}
