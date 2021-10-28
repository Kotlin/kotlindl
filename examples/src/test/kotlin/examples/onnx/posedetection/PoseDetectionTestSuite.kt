/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection

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

class PoseDetectionTestSuite {
    @Test
    fun easyPoseDetectionMoveNetSinglePoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

        model.use {
            for (i in 0..8) {
                val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
                // TODO
               /* val landmarks = it.detectPoseLandmarks(imageFile = imageFile)
                assertEquals(17, landmarks.size)
                val edges = it.detectPoseEdges(imageFile = imageFile)
                assertEquals(17, edges.size)*/
            }
        }
    }

    @Test
    fun easyPoseDetectionMoveNetSinglePoseThunderTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseEstimation.MoveNetSinglePoseThunder.pretrainedModel(modelHub)

        model.use {
            for (i in 0..8) {
                val imageFile = getFileFromResource("datasets/faces/image$i.jpg")
                // TODO
                /* val landmarks = it.detectPoseLandmarks(imageFile = imageFile)
                 assertEquals(17, landmarks.size)
                 val edges = it.detectPoseEdges(imageFile = imageFile)
                 assertEquals(17, edges.size)*/
            }
        }
    }

    @Test
    fun poseDetectionMoveNetSinglePoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting
        val model = modelHub.loadModel(modelType)

        model.use {
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = imageFile
                    imageShape = ImageShape(null, null, 3)
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

            val rawPoseLandMarks = (yhat as List<Array<Array<Array<FloatArray>>>>)[0][0][0]

            assertEquals(17, rawPoseLandMarks.size)
        }
    }

    @Test
    fun poseDetectionMoveNetSinglePoseThunderTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseEstimation.MoveNetSinglePoseThunder
        val model = modelHub.loadModel(modelType)

        model.use {
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = imageFile
                    imageShape = ImageShape(null, null, 3)
                }
                transformImage {
                    resize {
                        outputHeight = 256
                        outputWidth = 256
                    }
                    convert { colorMode = ColorMode.BGR }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)

            val rawPoseLandMarks = (yhat as List<Array<Array<Array<FloatArray>>>>)[0][0][0]

            assertEquals(17, rawPoseLandMarks.size)
        }
    }
}
