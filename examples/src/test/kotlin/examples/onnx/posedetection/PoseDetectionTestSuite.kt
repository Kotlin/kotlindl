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

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val detectedPose = poseDetectionModel.detectPose(imageFile = imageFile)
            assertEquals(17, detectedPose.poseLandmarks.size)
            assertEquals(18, detectedPose.edges.size)
        }
    }

    @Test
    fun easyPoseDetectionMoveNetSinglePoseThunderTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/single/1.jpg")
            val detectedPose = poseDetectionModel.detectPose(imageFile = imageFile)
            assertEquals(17, detectedPose.poseLandmarks.size)
            assertEquals(18, detectedPose.edges.size)
        }
    }

    @Test
    fun easyPoseDetectionMoveNetMultiPoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = ONNXModels.PoseEstimation.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

        model.use { poseDetectionModel ->
            val imageFile = getFileFromResource("datasets/poses/multi/1.jpg")
            val detectedPoses = poseDetectionModel.detectPoses(imageFile = imageFile)
            assertEquals(3, detectedPoses.multiplePoses.size)
            detectedPoses.multiplePoses.forEach {
                assertEquals(17, it.second.poseLandmarks.size)
                assertEquals(18, it.second.edges.size)
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

            val rawPoseLandMarks = (yhat["output_0"] as Array<Array<Array<FloatArray>>>)[0][0]

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

            val rawPoseLandMarks = (yhat["output_0"] as Array<Array<Array<FloatArray>>>)[0][0]

            assertEquals(17, rawPoseLandMarks.size)
        }
    }

    @Test
    fun poseDetectionMoveNetMultiPoseLightingTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.PoseEstimation.MoveNetMultiPoseLighting
        val model = modelHub.loadModel(modelType)

        model.use { inferenceModel ->
            val imageFile = getFileFromResource("datasets/poses/multi/1.jpg")
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
            val yhat = inferenceModel.predictRaw(inputData)
            println(yhat.values.toTypedArray().contentDeepToString())

            val rawPosesLandMarks = (yhat["output_0"] as Array<Array<FloatArray>>)[0]

            assertEquals(6, rawPosesLandMarks.size)
            rawPosesLandMarks.forEach {
                assertEquals(56, it.size)
            }
        }
    }
}
