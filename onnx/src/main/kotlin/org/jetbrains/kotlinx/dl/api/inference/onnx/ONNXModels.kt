/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.*
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDMobileNetV1ObjectDetectionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.MultiPoseDetectionModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.SinglePoseDetectionModel
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Transpose

/** Models in the ONNX format and running via ONNX Runtime. */
public object ONNXModels {
    /** Image recognition models and preprocessing. */
    public sealed class CV<T : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean // TODO: add to all models
    ) :
        ModelType<T, ImageRecognitionModel> {
        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return ImageRecognitionModel(modelHub.loadModel(this), this)
        }

        /** */
        public object ResNet18 : CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet18-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet34 : CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet34-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet50 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet50-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet101 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet101-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet152 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet152-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet18v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet18-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet34v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet34-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet50v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet50-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet101v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet101-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object ResNet152v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet152-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public object DenseNet121 :
            CV<OnnxInferenceModel>("models/onnx/cv/densenet/densenet121", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        }

        /** */
        public object EfficientNet4Lite :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-lite4", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(
                    data,
                    tensorShape,
                    inputType = InputType.TF,
                    channelsLast = false
                )
            }
        }

        /** */
        public object ResNet50custom :
            CV<OnnxInferenceModel>("models/onnx/cv/custom/resnet50", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(
                    data,
                    tensorShape,
                    inputType = InputType.CAFFE
                )
            }
        }

        /** */
        public object ResNet50noTopCustom :
            CV<OnnxInferenceModel>("models/onnx/cv/custom/resnet50notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(
                    data,
                    tensorShape,
                    inputType = InputType.CAFFE
                )
            }
        }

        /** */
        public object EfficientNetB0 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b0", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB0noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b0-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB1 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b1", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB1noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b1-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB2 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b2", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB2noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b2-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB3 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b3", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB3noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b3-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB4 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b4", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB4noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b4-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB5 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b5", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB5noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b5-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB6 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b6", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB6noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b6-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB7 :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b7", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object EfficientNetB7noTop :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b7-notop", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public object Lenet : CV<OnnxInferenceModel>("models/onnx/cv/custom/mnist", channelsFirst = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        }
    }

    /** Object detection models and preprocessing. */
    public sealed class ObjectDetection<T : InferenceModel, U : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean = true
    ) :
        ModelType<T, U> {
        /**
         * This model is a real-time neural network for object detection that detects 80 different classes.
         *
         * Image shape is (1x3x1200x1200).
         *
         * The model has 3 outputs:
         *  - boxes: (1x'nbox'x4)
         *  - labels: (1x'nbox')
         *  - scores: (1x'nbox')
         *
         * @see <a href="https://arxiv.org/abs/1512.02325">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd">
         *    Detailed description of SSD model and its pre- and postprocessing in onnx/models repository.</a>
         */
        public object SSD :
            ObjectDetection<OnnxInferenceModel, SSDObjectDetectionModel>("models/onnx/objectdetection/ssd") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                val transposedData = Transpose(axes = intArrayOf(2, 0, 1)).apply(
                    data,
                    ImageShape(width = tensorShape[0], height = tensorShape[1], channels = tensorShape[2])
                )

                // TODO: should be returned from the Transpose from apply method
                val transposedShape = longArrayOf(tensorShape[2], tensorShape[0], tensorShape[1])

                return preprocessInput(
                    transposedData,
                    transposedShape,
                    inputType = InputType.TORCH,
                    channelsLast = false
                )
            }

            override fun pretrainedModel(modelHub: ModelHub): SSDObjectDetectionModel {
                return modelHub.loadModel(this) as SSDObjectDetectionModel
            }
        }

        public object SSDMobileNetV1 :
            ObjectDetection<OnnxInferenceModel, SSDMobileNetV1ObjectDetectionModel>("models/onnx/objectdetection/ssd_mobilenet_v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
                /*return preprocessInput(
                    data,
                    tensorShape,
                    inputType = InputType.TORCH,
                    channelsLast = true
                )*/
            }

            override fun pretrainedModel(modelHub: ModelHub): SSDMobileNetV1ObjectDetectionModel {
                return modelHub.loadModel(this) as SSDMobileNetV1ObjectDetectionModel
            }
        }

        public object EfficientDetD2 :
            ObjectDetection<OnnxInferenceModel, SSDMobileNetV1ObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SSDMobileNetV1ObjectDetectionModel {
                return modelHub.loadModel(this) as SSDMobileNetV1ObjectDetectionModel
            }
        }

        /** */
        public object YOLOv4 :
            ObjectDetection<OnnxInferenceModel, OnnxInferenceModel>("models/onnx/objectdetection/yolov4") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }

            override fun pretrainedModel(modelHub: ModelHub): OnnxInferenceModel {
                TODO("Not yet implemented")
            }
        }
    }

    /** Face alignment models and preprocessing. */
    public sealed class FaceAlignment<T : InferenceModel, U : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean = true
    ) :
        ModelType<T, U> {
        /** */
        public object Fan2d106 :
            FaceAlignment<OnnxInferenceModel, Fan2D106FaceAlignmentModel>("models/onnx/facealignment/fan_2d_106") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                val transposedData = Transpose(axes = intArrayOf(2, 0, 1)).apply(
                    data,
                    ImageShape(width = tensorShape[0], height = tensorShape[1], channels = tensorShape[2])
                )

                // TODO: should be returned from the Transpose from apply method
                val transposedShape = longArrayOf(tensorShape[2], tensorShape[0], tensorShape[1])

                return transposedData
            }

            override fun pretrainedModel(modelHub: ModelHub): Fan2D106FaceAlignmentModel {
                return Fan2D106FaceAlignmentModel(modelHub.loadModel(this))
            }
        }
    }

    /** Face alignment models and preprocessing. */
    public sealed class PoseEstimation<T : InferenceModel, U : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean = true
    ) :
        ModelType<T, U> {
        /** */
        public object MoveNetSinglePoseLighting :
            PoseEstimation<OnnxInferenceModel, SinglePoseDetectionModel>("models/onnx/poseestimation/movenet_singlepose_lighting_13") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return modelHub.loadModel(this) as SinglePoseDetectionModel
            }
        }

        /**
         *
         *   The ```predictRaw``` method returns a float32 tensor of shape (1, 6, 56).
         *
         *   - The first dimension is the batch dimension, which is always equal to 1.
         *   - The second dimension corresponds to the maximum number of instance detections.
         *   - The model can detect up to 6 people in the image frame simultaneously.
         *   - The third dimension represents the predicted bounding box/keypoint locations and scores.
         *   - The first 17 * 3 elements are the keypoint locations and scores in the format: ```[y_0, x_0, s_0, y_1, x_1, s_1, …, y_16, x_16, s_16]```,
         *     where y_i, x_i, s_i are the yx-coordinates (normalized to image frame, e.g. range in ```[0.0, 1.0]```) and confidence scores of the i-th joint correspondingly.
         *   - The order of the 17 keypoint joints is: ```[nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]```.
         *   - The remaining 5 elements ```[ymin, xmin, ymax, xmax, score]``` represent the region of the bounding box (in normalized coordinates) and the confidence score of the instance.
         */
        public object MoveNetMultiPoseLighting :
            PoseEstimation<OnnxInferenceModel, MultiPoseDetectionModel>("models/onnx/poseestimation/movenet_multipose_lighting") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): MultiPoseDetectionModel {
                return modelHub.loadModel(this) as MultiPoseDetectionModel
            }
        }

        /** */
        public object MoveNetSinglePoseThunder :
            PoseEstimation<OnnxInferenceModel, SinglePoseDetectionModel>("models/onnx/poseestimation/movenet_thunder") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return modelHub.loadModel(this) as SinglePoseDetectionModel
            }
        }
    }
}

internal fun resNetOnnxPreprocessing(data: FloatArray, tensorShape: LongArray): FloatArray {
    val transposedData = Transpose(axes = intArrayOf(2, 0, 1)).apply(
        data,
        ImageShape(width = tensorShape[0], height = tensorShape[1], channels = tensorShape[2])
    )

    // TODO: should be returned from the Transpose from apply method
    val transposedShape = longArrayOf(tensorShape[2], tensorShape[0], tensorShape[1])

    return preprocessInput(
        transposedData,
        transposedShape,
        inputType = InputType.TF,
        channelsLast = false
    )
}
