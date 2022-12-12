/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.File

class FacesTestSuite {
    @Test
    fun easyFan2D106Test() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)

        model.use {
            for (i in 0..8) {
                val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
                val landmarks = it.detectLandmarks(imageFile = imageFile)
                assertEquals(106, landmarks.size)
            }
        }
    }

    @Test
    fun fan2D106Test() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.FaceAlignment.Fan2d106
        val model = modelHub.loadModel(modelType)

        model.use {
            val fileDataLoader = pipeline<BufferedImage>()
                .resize {
                    outputHeight = 192
                    outputWidth = 192
                }
                .convert { colorMode = ColorMode.BGR }
                .toFloatArray { }
                .call(modelType.preprocessor)
                .fileLoader()

            for (i in 0..8) {
                val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
                val inputData = fileDataLoader.load(imageFile)

                val yhat = it.predictRaw(inputData) { output -> output.getFloatArray(0) }
                assertEquals(212, yhat.size)
            }
        }
    }
}
