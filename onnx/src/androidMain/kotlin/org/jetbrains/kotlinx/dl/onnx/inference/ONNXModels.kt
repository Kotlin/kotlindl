/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.normalize
import org.jetbrains.kotlinx.dl.impl.preprocessing.rescale
import org.jetbrains.kotlinx.dl.onnx.inference.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.FaceDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDLikeModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDLikeModelMetadata
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.SinglePoseDetectionModel

/**
 * Set of pretrained mobile-friendly ONNX models
 */
public object ONNXModels {
    /** Image classification models.
     *
     * @property [channelsFirst] If true it means that the second dimension is related to number of channels in image
     *                           has short notation as `NCWH`,
     *                           otherwise, channels are at the last position and has a short notation as `NHWC`.
     * */
    public sealed class CV(
        override val modelRelativePath: String,
        protected val channelsFirst: Boolean
    ) : OnnxModelType<ImageRecognitionModel> {
        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return ImageRecognitionModel(
                modelHub.loadModel(this),
                channelsFirst,
                preprocessor,
                this::class.simpleName
            )
        }

        /**
         * Image classification model based on EfficientNet-Lite architecture.
         * Trained on ImageNet 1k dataset.
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.labels] method).
         *
         * EfficientNet-Lite 4 is the largest variant and most accurate of the set of EfficientNet-Lite model.
         * It is an integer-only quantized model that produces the highest accuracy of all the EfficientNet models.
         * It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * @see <a href="https://arxiv.org/abs/1905.11946">
         *     EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a>
         * @see <a href="https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4">
         *    Official EfficientNet4Lite model from ONNX Github.</a>
         */
        public object EfficientNet4Lite : CV("efficientnet_lite4", channelsFirst = false) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing(channelsLast = !channelsFirst)
        }

        /**
         * Image classification model based on MobileNetV1 architecture.
         * Trained on ImageNet 1k dataset.
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.labels] method).
         *
         * MobileNetV1 is small, low-latency, low-power model and can be run efficiently on mobile devices
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1001)
         *
         * @see <a href="https://arxiv.org/abs/1905.11946">
         *     EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a>
         * @see <a href="https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4">
         *    Official EfficientNet4Lite model from ONNX Github.</a>
         */
        public object MobilenetV1 : CV("mobilenet_v1", channelsFirst = false) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = pipeline<FloatData>()
                    .rescale { scalingCoefficient = 255f }
                    .normalize {
                        mean = floatArrayOf(0.5f, 0.5f, 0.5f)
                        std = floatArrayOf(0.5f, 0.5f, 0.5f)
                        channelsLast = !channelsFirst
                    }

            override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
                return ImageRecognitionModel(
                    modelHub.loadModel(this),
                    channelsFirst,
                    preprocessor,
                    this::class.simpleName,
                    classLabels = Imagenet.V1001.labels()
                )
            }
        }
    }

    /** Pose detection models. */
    public sealed class PoseDetection<U : InferenceModel>(override val modelRelativePath: String) :
        OnnxModelType<U> {
        /**
         * This model is a convolutional neural network model that runs on RGB images and predicts human joint locations of a single person.
         * (edges are available in [org.jetbrains.kotlinx.dl.onnx.inference.posedetection.edgeKeyPointsPairs]
         * and keypoints are in [org.jetbrains.kotlinx.dl.onnx.inference.posedetection.keyPoints]).
         *
         * Model architecture: MobileNetV2 image feature extractor with Feature Pyramid Network decoder (to stride of 4)
         * followed by CenterNet prediction heads with custom post-processing logics. Lightning uses depth multiplier 1.0.
         *
         * The model have an input tensor with type INT32 and shape `[1, 192, 192, 3]`.
         *
         * The model has 1 output:
         * - output_0 tensor with type FLOAT32 and shape `[1, 1, 17, 3]` with 17 rows related to the following keypoints
         * `[nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]`.
         * Each row contains 3 numbers: `[y, x, confidence_score]` normalized in `[0.0, 1.0]` range.
         *
         * @see <a href="https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html">
         *     Detailed description of MoveNet architecture in TensorFlow blog.</a>
         * @see <a href="https://tfhub.dev/google/movenet/singlepose/lightning/4">
         *    TensorFlow Model Hub with the MoveNetLighting model converted to ONNX.</a>
         */
        public object MoveNetSinglePoseLighting :
            PoseDetection<SinglePoseDetectionModel>("movenet_singlepose_lighting_13") {
            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return SinglePoseDetectionModel(modelHub.loadModel(this), this::class.simpleName)
            }
        }

        /**
         * This model is a convolutional neural network model that runs on RGB images and predicts human joint locations of a single person.
         * (edges are available in [org.jetbrains.kotlinx.dl.onnx.inference.posedetection.edgeKeyPointsPairs]
         * and keypoints are in [org.jetbrains.kotlinx.dl.onnx.inference.posedetection.keyPoints]).
         *
         * Model architecture: MobileNetV2 image feature extractor with Feature Pyramid Network decoder (to stride of 4)
         * followed by CenterNet prediction heads with custom post-processing logics. Lightning uses depth multiplier 1.0.
         *
         * The model have an input tensor with type INT32 and shape `[1, 192, 192, 3]`.
         *
         * The model has 1 output:
         * - output_0 tensor with type FLOAT32 and shape `[1, 1, 17, 3]` with 17 rows related to the following keypoints
         * `[nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]`.
         * Each row contains 3 numbers: `[y, x, confidence_score]` normalized in `[0.0, 1.0]` range.
         *
         * @see <a href="https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html">
         *     Detailed description of MoveNet architecture in TensorFlow blog.</a>
         * @see <a href="https://tfhub.dev/google/movenet/singlepose/thunder/4">
         *    TensorFlow Model Hub with the MoveNetLighting model converted to ONNX.</a>
         */
        public object MoveNetSinglePoseThunder :
            PoseDetection<SinglePoseDetectionModel>("movenet_thunder") {
            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return SinglePoseDetectionModel(modelHub.loadModel(this), this::class.simpleName)
            }
        }
    }

    /** Object detection models and preprocessing. */
    public sealed class ObjectDetection<U : InferenceModel>(override val modelRelativePath: String) :
        OnnxModelType<U> {
        /**
         * This model is a real-time neural network for object detection that detects 90 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.impl.dataset.Coco.V2017]).
         *
         * SSD-MobilenetV1 is an object detection model that uses a Single Shot MultiBox Detector (SSD) approach
         * to predict object classes for boundary boxes.
         *
         * SSD is a CNN that enables the model to only need to take one single shot to detect multiple objects in an image,
         * and MobileNet is a CNN base network that provides high-level features for object detection.
         * The combination of these two model frameworks produces an efficient,
         * high-accuracy detection model that requires less computational cost.
         *
         * The model have an input with the shape is (1x300x300x3).
         *
         * The model has 4 outputs:
         * - num_detections: the number of detections.
         * - detection_boxes: a list of bounding boxes. Each list item describes a box with top, left, bottom, right relative to the image size.
         * - detection_scores: the score for each detection with values between 0 and 1 representing probability that a class was detected.
         * - detection_classes: Array of 10 integers (floating point values) indicating the index of a class label from the COCO class.
         *
         * @see <a href="https://arxiv.org/abs/1512.02325">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd-mobilenetv1">
         *    Detailed description of SSD model and its pre- and postprocessing in onnx/models repository.</a>
         */
        public object SSDMobileNetV1 :
            ObjectDetection<SSDLikeModel>("ssd_mobilenet_v1") {

            private val METADATA = SSDLikeModelMetadata(
                "TFLite_Detection_PostProcess",
                "TFLite_Detection_PostProcess:1",
                "TFLite_Detection_PostProcess:2",
                0, 1
            )

            override fun pretrainedModel(modelHub: ModelHub): SSDLikeModel {
                return SSDLikeModel(modelHub.loadModel(this), METADATA, this::class.simpleName)
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 90 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.impl.dataset.Coco.V2017]).
         *
         * Internally it uses the EfficientNetLite as backbone network.
         *
         * The model have an input with the shape is (1x320x320x3).
         *
         * The model has 4 outputs:
         * - num_detections: the number of detections.
         * - detection_boxes: a list of bounding boxes. Each list item describes a box with top, left, bottom, right relative to the image size.
         * - detection_scores: the score for each detection with values between 0 and 1 representing probability that a class was detected.
         * - detection_classes: Array of 10 integers (floating point values) indicating the index of a class label from the COCO class.
         *
         * NOTE: The detections are limited to 25.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     EfficientDet: Scalable and Efficient Object Detection.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetLite0 :
            ObjectDetection<SSDLikeModel>("efficientdet_lite0") {

            private val METADATA = SSDLikeModelMetadata(
                "StatefulPartitionedCall:3",
                "StatefulPartitionedCall:2",
                "StatefulPartitionedCall:1",
                0, 1
            )

            override fun pretrainedModel(modelHub: ModelHub): SSDLikeModel {
                return SSDLikeModel(modelHub.loadModel(this), METADATA, this::class.simpleName)
            }
        }
    }

    /** Face detection models */
    public sealed class FaceDetection(override val modelRelativePath: String) :
        OnnxModelType<FaceDetectionModel> {
        override val preprocessor: Operation<FloatData, FloatData>
            get() = defaultPreprocessor

        override fun pretrainedModel(modelHub: ModelHub): FaceDetectionModel {
            return FaceDetectionModel(modelHub.loadModel(this), this::class.simpleName)
        }

        /**
         * Ultra-lightweight face detection model.
         *
         * Model accepts input of the shape (1 x 3 x 240 x 320)
         * Model outputs two arrays (1 x 4420 x 2) and (1 x 4420 x 4) of scores and boxes.
         *
         * Threshold filtration and non-max suppression are applied during postprocessing.
         *
         * @see <a href="https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface">Ultra-lightweight face detection model</a>
         */
        public object UltraFace320 : FaceDetection("ultraface_320")

        /**
         * Ultra-lightweight face detection model.
         *
         * Model accepts input of the shape (1 x 3 x 480 x 640)
         * Model outputs two arrays (1 x 4420 x 2) and (1 x 4420 x 4) of scores and boxes.
         *
         * Threshold filtration and non-max suppression are applied during postprocessing.
         *
         * @see <a href="https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface">Ultra-lightweight face detection model</a>
         */
        public object UltraFace640 : FaceDetection("ultraface_640")

        public companion object {
            public val defaultPreprocessor: Operation<FloatData, FloatData> =
                pipeline<FloatData>()
                    .normalize {
                        mean = floatArrayOf(127f, 127f, 127f)
                        std = floatArrayOf(128f, 128f, 128f)
                        channelsLast = false
                    }
        }
    }

    /** Face alignment models */
    public sealed class FaceAlignment<U : InferenceModel> : OnnxModelType<U> {
        /**
         * This model is a neural network for face alignment that take RGB images of faces as input and produces coordinates of 106 faces landmarks.
         *
         * The model have
         * - an input with the shape (1x3x192x192)
         * - an output with the shape (1x212)
         */
        public object Fan2d106 : FaceAlignment<Fan2D106FaceAlignmentModel>() {
            override val modelRelativePath: String = "fan_2d_106"
            override fun pretrainedModel(modelHub: ModelHub): Fan2D106FaceAlignmentModel {
                return Fan2D106FaceAlignmentModel(modelHub.loadModel(this), this::class.simpleName)
            }
        }
    }
}
