/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.resnet

import examples.transferlearning.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels

fun resnet50Prediction() {
    runImageRecognitionPrediction(modelType = TFModels.CV.ResNet50())
}

/** */
fun main(): Unit = resnet50Prediction()
