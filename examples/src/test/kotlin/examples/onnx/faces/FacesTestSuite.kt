/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.faces

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
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
            for (i in 0..8) {
                val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
                val preprocessing: Preprocessing = preprocess {
                    load {
                        pathToData = imageFile
                        imageShape = ImageShape(224, 224, 3)
                    }
                    transformImage {
                        resize {
                            outputHeight = 192
                            outputWidth = 192
                        }
                        convert { colorMode = ColorMode.BGR }
                    }
                }

                val inputData = modelType.preprocessInput(preprocessing)

                val yhat = it.predictRaw(inputData)
                assertEquals(212, (yhat.values.toTypedArray()[0] as Array<FloatArray>)[0].size)
            }
        }
    }
}
