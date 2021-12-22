/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.*
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.EfficientDetObjectDetectionModel
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
        override val channelsFirst: Boolean,
        internal var noTop: Boolean = false
    ) :
        ModelType<T, ImageRecognitionModel> {
        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return ImageRecognitionModel(modelHub.loadModel(this), this)
        }

        override fun preInit(): InferenceModel {
            return OnnxInferenceModel()
        }

        /** */
        public class ResNet18 : CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet18-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet34 : CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet34-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet50 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet50-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet101 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet101-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet152 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet152-v1", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet18v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet18-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet34v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet34-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet50v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet50-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet101v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet101-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class ResNet152v2 :
            CV<OnnxInferenceModel>("models/onnx/cv/resnet/resnet152-v2", channelsFirst = true) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return resNetOnnxPreprocessing(data, tensorShape)
            }
        }

        /** */
        public class EfficientNet4Lite :
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
        public class EfficientNetB0(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b0", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB1(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b1", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB2(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b2", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB3(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b3", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB4(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b4", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB5(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b5", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB6(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b6", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class EfficientNetB7(noTop: Boolean = false) :
            CV<OnnxInferenceModel>("models/onnx/cv/efficientnet/efficientnet-b7", channelsFirst = false, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }
        }

        /** */
        public class Lenet : CV<OnnxInferenceModel>("models/onnx/cv/custom/mnist", channelsFirst = false) {
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
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories]).
         *
         * The model have an input with the shape is (1x3x1200x1200).
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

            override fun preInit(): SSDObjectDetectionModel {
                return SSDObjectDetectionModel()
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories]).
         *
         * SSD-MobilenetV1 is an object detection model that uses a Single Shot MultiBox Detector (SSD) approach
         * to predict object classes for boundary boxes.
         *
         * SSD is a CNN that enables the model to only need to take one single shot to detect multiple objects in an image,
         * and MobileNet is a CNN base network that provides high-level features for object detection.
         * The combination of these two model frameworks produces an efficient,
         * high-accuracy detection model that requires less computational cost.
         *
         * The model have an input with the shape is (1xHxWx3). H and W could be defined by user. H = W = 1000 by default.
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
            ObjectDetection<OnnxInferenceModel, SSDMobileNetV1ObjectDetectionModel>("models/onnx/objectdetection/ssd_mobilenet_v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SSDMobileNetV1ObjectDetectionModel {
                return modelHub.loadModel(this) as SSDMobileNetV1ObjectDetectionModel
            }

            override fun preInit(): SSDMobileNetV1ObjectDetectionModel {
                val model =  SSDMobileNetV1ObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 1000L, 1000L, 3L)
                return model
            }
        }

        // TODO: download from here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md#tensorflow-2-detection-model-zoo
        // TODO: interesting visualisation https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
        // TODO: good tutorial https://www.tensorflow.org/hub/tutorials/tf2_object_detection
        // TODO: looks like the right models were exported https://github.com/google/automl/tree/master/efficientdet


        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x512x512x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD0 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d0") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 512L, 512L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (640x640x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD1 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 640L, 640L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x768x768x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD2 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 768L, 768L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x896x896x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD3 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d3") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 896L, 896L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x1024x1024x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD4 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d4") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 1024L, 1024L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x1280x1280x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD5 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d5") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 1280L, 1280L, 3L)
                return model
            }
        }

        /**
         * This model is a real-time neural network for object detection that detects 80 different classes
         * (labels are available in [org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForEfficientDet]).
         *
         * Internally it uses the EfficientNets as backbone networks.
         *
         * The model have an input with the shape is (1x1280x1280x3) by default. H and W could be changed by user to any values.
         *
         * The model has 1 output:
         * - detections:0 with 7 numbers as `[unknown number, ymin, _xmin_, ymax, xmax, score, coco label]`.
         *
         * NOTE: The detections are limited to 100.
         *
         * @see <a href="https://arxiv.org/abs/1911.09070">
         *     SSD: Single Shot MultiBox Detector.</a>
         * @see <a href="https://github.com/google/automl/tree/master/efficientdet">
         *    Detailed description of EfficientDet model in google/automl repository.</a>
         * @see <a href="https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb">
         *    Tutorial which shows how to covert the EfficientDet models to ONNX using tf2onnx.</a>
         */
        public object EfficientDetD6 :
            ObjectDetection<OnnxInferenceModel, EfficientDetObjectDetectionModel>("models/onnx/objectdetection/efficientdet/efficientdet-d6") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): EfficientDetObjectDetectionModel {
                return modelHub.loadModel(this) as EfficientDetObjectDetectionModel
            }

            override fun preInit(): EfficientDetObjectDetectionModel {
                val model =  EfficientDetObjectDetectionModel()
                model.inputShape = longArrayOf(1L, 1280L, 1280L, 3L)
                return model
            }
        }

        /** */
        // TODO: remove or implement
        public object YOLOv4 :
            ObjectDetection<OnnxInferenceModel, OnnxInferenceModel>("models/onnx/objectdetection/yolov4") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }

            override fun pretrainedModel(modelHub: ModelHub): OnnxInferenceModel {
                TODO("Not yet implemented")
            }

            override fun preInit(): OnnxInferenceModel {
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

            override fun preInit(): OnnxInferenceModel {
                return OnnxInferenceModel()
            }
        }
    }

    /** Pose estimation models. */
    public sealed class PoseEstimation<T : InferenceModel, U : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean = true
    ) :
        ModelType<T, U> {
        /**
         * This model is a convolutional neural network model that runs on RGB images and predicts human joint locations of a single person.
         * (edges are available in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.edgeKeyPointsPairs]
         * and keypoints are in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.keyPoints]).
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
            PoseEstimation<OnnxInferenceModel, SinglePoseDetectionModel>("models/onnx/poseestimation/movenet_singlepose_lighting_13") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return modelHub.loadModel(this) as SinglePoseDetectionModel
            }

            override fun preInit(): SinglePoseDetectionModel {
                return SinglePoseDetectionModel()
            }
        }

        /**
         * A convolutional neural network model that runs on RGB images and predicts human joint locations of people in the image frame.
         * The main differentiator between this MoveNet.MultiPose and its precedent, MoveNet.SinglePose model,
         * is that this model is able to detect multiple people in the image frame at the same time while still achieving real-time speed.
         *
         * (edges are available in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.edgeKeyPointsPairs]
         * and keypoints are in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.keyPoints]).
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
         *
         * @see <a href="https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html">
         *     Detailed description of MoveNet architecture in TensorFlow blog.</a>
         * @see <a href="https://tfhub.dev/google/movenet/multipose/lightning/1">
         *    TensorFlow Model Hub with the MoveNetLighting model converted to ONNX.</a>
         */
        public object MoveNetMultiPoseLighting :
            PoseEstimation<OnnxInferenceModel, MultiPoseDetectionModel>("models/onnx/poseestimation/movenet_multipose_lighting") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): MultiPoseDetectionModel {
                return modelHub.loadModel(this) as MultiPoseDetectionModel
            }

            override fun preInit(): MultiPoseDetectionModel {
                val model = MultiPoseDetectionModel()
                model.inputShape = longArrayOf(1L, 256L, 256L, 3L)
                return model
            }
        }

        /**
         * This model is a convolutional neural network model that runs on RGB images and predicts human joint locations of a single person.
         * (edges are available in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.edgeKeyPointsPairs]
         * and keypoints are in [org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection.keyPoints]).
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
            PoseEstimation<OnnxInferenceModel, SinglePoseDetectionModel>("models/onnx/poseestimation/movenet_thunder") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return data
            }

            override fun pretrainedModel(modelHub: ModelHub): SinglePoseDetectionModel {
                return modelHub.loadModel(this) as SinglePoseDetectionModel
            }

            override fun preInit(): SinglePoseDetectionModel {
                return SinglePoseDetectionModel()
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
