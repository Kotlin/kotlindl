/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection

import examples.onnx.objectdetection.ssd.ssdLightAPI
import examples.onnx.objectdetection.ssd.ssd
import examples.onnx.objectdetection.ssdmobile.ssdMobile
import examples.onnx.objectdetection.ssdmobile.ssdMobileLightAPI
import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.EfficientDetObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
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
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD0, 51)
    }

    @Test
    fun efficientDetD1Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD1)
    }

    @Test
    fun efficientDetD1LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD1, 50)
    }

    @Test
    fun efficientDetD2Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD2)
    }

    @Test
    fun efficientDetD2LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD2, 38)
    }

    @Test
    fun efficientDetD3Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD3)
    }

    @Test
    fun efficientDetD3LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD3, 33)
    }

    @Test
    fun efficientDetD4Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD4)
    }

    @Test
    fun efficientDetD4LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD4, 27)
    }

    @Test
    fun efficientDetD5Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD5)
    }

    @Test
    fun efficientDetD5LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD5, 24)
    }

    @Test
    fun efficientDetD6Test() {
        efficientDetInference(ONNXModels.ObjectDetection.EfficientDetD6)
    }

    @Test
    fun efficientDetD6LightAPITest() {
        efficientDetLightAPIInference(ONNXModels.ObjectDetection.EfficientDetD6, 19)
    }
}


fun efficientDetInference(modelType: ONNXModels.ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    model.use {
        for (i in 1..6) {
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = getFileFromResource("datasets/detection/image$i.jpg")
                    imageShape = ImageShape(null, null, 3)
                }
                transformImage {
                    resize {
                        outputHeight = it.inputShape[1].toInt()
                        outputWidth = it.inputShape[2].toInt()
                    }
                    convert { colorMode = ColorMode.BGR }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)
            assertTrue { yhat.containsKey("detections:0") }
        }
    }
}

fun efficientDetLightAPIInference(modelType: ONNXModels.ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>, numberOfDetectedObjects: Int) {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadPretrainedModel(modelType)

    model.use { detectionModel ->
        val imageFile = getFileFromResource("datasets/detection/image4.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile)

        assertEquals(numberOfDetectedObjects, detectedObjects.size)
    }
}
