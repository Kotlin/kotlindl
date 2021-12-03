/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.*
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
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
