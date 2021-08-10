/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel

private const val PATH_TO_MODEL_1 = "examples/src/main/resources/models/onnx/mnist-8.onnx"
private const val PATH_TO_MODEL_2 = "examples/src/main/resources/models/onnx/resnet50.onnx"
private const val PATH_TO_MODEL_3 = "examples/src/main/resources/models/onnx/resnet50notop.onnx"
private const val PATH_TO_MODEL_4 = "examples/src/main/resources/models/onnx/efficientnet-lite4-11.onnx"


fun main() {
    var model = OnnxInferenceModel.load(PATH_TO_MODEL_1)

    println(model.toString())

    model.close()

    model = OnnxInferenceModel.load(PATH_TO_MODEL_2)

    println(model.toString())

    model.close()

    model = OnnxInferenceModel.load(PATH_TO_MODEL_3)

    println(model.toString())

    model.close()

    model = OnnxInferenceModel.load(PATH_TO_MODEL_4)

    println(model.toString())

    model.close()
}
