/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv.resnet

import examples.onnx.cv.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels

/**
 * This examples demonstrates the inference concept on ResNet'18 model:
 * - Model configuration, model weights and labels are obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in ResNet'18 during training on ImageNet dataset) is applied to each image before prediction.
 */
fun resnet18prediction() {
    runImageRecognitionPrediction(ONNXModels.CV.ResNet18)
}

/** */
fun main(): Unit = resnet18prediction()
