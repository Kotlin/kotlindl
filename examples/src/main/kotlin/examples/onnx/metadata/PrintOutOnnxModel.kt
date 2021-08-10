/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxModel

private const val PATH_TO_MODEL_1 = "examples/src/main/resources/models/onnx/mnist-8.onnx"
private const val PATH_TO_MODEL_2 = "examples/src/main/resources/models/onnx/resnet50.onnx"
private const val PATH_TO_MODEL_3 = "examples/src/main/resources/models/onnx/resnet50notop.onnx"

fun main() {
    var model = OnnxModel.load(PATH_TO_MODEL_1)

    println(model.toString())

    model.close()

    model = OnnxModel.load(PATH_TO_MODEL_2)

    println(model.toString())

    model.close()

    model = OnnxModel.load(PATH_TO_MODEL_3)

    println(model.toString())

    model.close()
}
