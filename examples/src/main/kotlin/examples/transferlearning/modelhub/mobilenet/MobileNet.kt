/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.mobilenet


import examples.transferlearning.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub

/**
 * This examples demonstrates the inference concept on MobileNet model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in MobileNet during training on ImageNet dataset) is applied to the images before prediction.
 */
fun mobileNetPrediction() {
    runImageRecognitionPrediction(modelType = TFModels.CV.MobileNet())
}

/** */
fun main(): Unit = mobileNetPrediction()
