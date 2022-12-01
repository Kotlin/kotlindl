/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.EfficientDetObjectDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.MultiPoseDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.SinglePoseDetectionModel
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.io.File

class ModelCopyingTestSuite {
    @Test
    fun efficientNetCopyTest() {
        runCopyTest(
            listOf(
                ONNXModels.CV.EfficientNetB0(),
                ONNXModels.CV.EfficientNetB1(),
                ONNXModels.CV.EfficientNetB2(),
                ONNXModels.CV.EfficientNetB3(),
                ONNXModels.CV.EfficientNetB4(),
                ONNXModels.CV.EfficientNetB5(),
                ONNXModels.CV.EfficientNetB6(),
                ONNXModels.CV.EfficientNetB7(),
                ONNXModels.CV.EfficientNet4Lite
            ),
            "datasets/vgg/image0.jpg",
            ImageRecognitionModel::predictObject
        )
    }

    @Test
    fun resNetCopyTest() {
        runCopyTest(
            listOf(
                ONNXModels.CV.ResNet18,
                ONNXModels.CV.ResNet18v2,
                ONNXModels.CV.ResNet34,
                ONNXModels.CV.ResNet34v2,
                ONNXModels.CV.ResNet50,
                ONNXModels.CV.ResNet50v2,
                ONNXModels.CV.ResNet50custom,
                ONNXModels.CV.ResNet101,
                ONNXModels.CV.ResNet101v2,
                ONNXModels.CV.ResNet152,
                ONNXModels.CV.ResNet152v2
            ),
            "datasets/vgg/image0.jpg",
            ImageRecognitionModel::predictObject
        )
    }

    @Test
    fun fan2D106CopyTest() {
        runCopyTest(
            listOf(ONNXModels.FaceAlignment.Fan2d106), "datasets/faces/image1.jpg",
            Fan2D106FaceAlignmentModel::detectLandmarks
        )
    }

    @Test
    fun efficientDetCopyTest() {
        runCopyTest(
            listOf(
                ONNXModels.ObjectDetection.EfficientDetD0,
                ONNXModels.ObjectDetection.EfficientDetD1,
                ONNXModels.ObjectDetection.EfficientDetD2,
                ONNXModels.ObjectDetection.EfficientDetD3,
                ONNXModels.ObjectDetection.EfficientDetD4,
                ONNXModels.ObjectDetection.EfficientDetD5,
                ONNXModels.ObjectDetection.EfficientDetD6
            ),
            "datasets/detection/image1.jpg",
            EfficientDetObjectDetectionModel::detectObjects
        )
    }

    @Test
    fun moveNetMultiPoseCopyTest() {
        runCopyTest(
            listOf(
                ONNXModels.PoseDetection.MoveNetMultiPoseLighting
            ), "datasets/poses/multi/1.jpg",
            MultiPoseDetectionModel::detectPoses
        )
    }

    @Test
    fun moveNetSinglePoseCopyTest() {
        runCopyTest(
            listOf(
                ONNXModels.PoseDetection.MoveNetSinglePoseLighting,
                ONNXModels.PoseDetection.MoveNetSinglePoseThunder
            ), "datasets/poses/single/1.jpg",
            SinglePoseDetectionModel::detectPose
        )
    }

    companion object {
        private fun <M : InferenceModel, R> runCopyTest(
            modelTypes: List<ModelType<OnnxInferenceModel, M>>,
            fileName: String,
            detect: M.(File) -> R
        ) {
            val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
            val imageFile = getFileFromResource(fileName)
            for (modelType in modelTypes) {
                modelHub.loadPretrainedModel(modelType, LoadingMode.SKIP_LOADING_IF_EXISTS).use { model ->
                    @Suppress("UNCHECKED_CAST")
                    (model.copy("model_copy") as M).use { modelCopy ->

                        val detectionResult = detect(model, imageFile)
                        val detectionResultCopy = detect(modelCopy, imageFile)

                        Assertions.assertEquals(detectionResult, detectionResultCopy) { "Copy failed for $modelType" }
                    }
                }
            }
        }
    }
}