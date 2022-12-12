/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection

import examples.onnx.objectdetection.ssd.ssd
import examples.onnx.objectdetection.ssd.ssdLightAPI
import examples.onnx.objectdetection.ssdmobile.ssdMobile
import examples.onnx.objectdetection.ssdmobile.ssdMobileLightAPI
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
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.EfficientDetObjectDetectionModel
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.File

class ObjectDetectionTestSuite {
    @Test
    fun ssdTest() {
        ssd()
    }

    @Test
    fun ssdLightAPITest() {
        ssdLightAPI()
    }

    @Test
    fun ssdMobileTest() {
        ssdMobile()
    }

    @Test
    fun ssdMobileLightAPITest() {
        ssdMobileLightAPI()
    }

    @Test
    fun efficientDetD0Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD0)
    }

    @Test
    fun efficientDetD0LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD0, 72)
    }

    @Test
    fun efficientDetD1Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD1)
    }

    @Test
    fun efficientDetD1LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD1, 65)
    }

    @Test
    fun efficientDetD2Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD2)
    }

    @Test
    fun efficientDetD2LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD2, 51)
    }

    @Test
    fun efficientDetD3Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD3)
    }

    @Test
    fun efficientDetD3LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD3, 38)
    }

    @Test
    fun efficientDetD4Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD4)
    }

    @Test
    fun efficientDetD4LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD4, 26)
    }

    @Test
    fun efficientDetD5Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD5)
    }

    @Test
    fun efficientDetD5LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD5, 26)
    }

    @Test
    fun efficientDetD6Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD6)
    }

    @Test
    fun efficientDetD6LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD6, 22)
    }
}


fun efficientDetInference(modelType: ONNXModels.ObjectDetection<EfficientDetObjectDetectionModel>) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    model.use {
        val fileDataLoader = pipeline<BufferedImage>()
            .resize {
                outputHeight = it.inputDimensions[0].toInt()
                outputWidth = it.inputDimensions[1].toInt()
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 1..6) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/detection/image$i.jpg"))

            val yhat = it.predictRaw(inputData)
            assertTrue { yhat.containsKey("detections:0") }
        }
    }
}

fun efficientDetLightAPIInference(
    modelType: ONNXModels.ObjectDetection<EfficientDetObjectDetectionModel>,
    numberOfDetectedObjects: Int
) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadPretrainedModel(modelType)

    model.use { detectionModel ->
        val imageFile = getFileFromResource("datasets/detection/image4.jpg")
        val detectedObjects = detectionModel.detectObjects(imageFile = imageFile, topK = 0)

        assertEquals(numberOfDetectedObjects, detectedObjects.size)
    }
}
