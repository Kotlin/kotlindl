/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx;

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.InputType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Transpose

public enum class ONNXModels {
    ;

    public enum class CV(override val modelRelativePath: String) : ModelType {
        /** */
        ResNet_18_v1("models/onnx/cv/resnet/resnet18-v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_34_v1("models/onnx/cv/resnet/resnet34-v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_50_v1("models/onnx/cv/resnet/resnet50-v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_101_v1("models/onnx/cv/resnet/resnet101-v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_152_v1("models/onnx/cv/resnet/resnet151-v1") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_18_v2("models/onnx/cv/resnet/resnet18-v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_34_v2("models/onnx/cv/resnet/resnet34-v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_50_v2("models/onnx/cv/resnet/resnet50-v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_101_v2("models/onnx/cv/resnet/resnet101-v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_151_v2("models/onnx/cv/resnet/resnet151-v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        DenseNet_121("models/onnx/cv/densenet/densenet121") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        EfficientNet_4_Lite("models/onnx/cv/efficientnet/efficientnet-lite4") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_50_v1_custom("models/onnx/cv/custom/resnet50") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        ResNet_50_v1_no_top_custom("models/onnx/cv/custom/resnet50notop") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },

        /** */
        Lenet_mnist("models/onnx/cv/custom/mnist") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        }
    }
    ;

    public enum class ObjectDetection(override val modelRelativePath: String) : ModelType {
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
        SSD("models/onnx/objectdetection/ssd") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                val processedData = org.jetbrains.kotlinx.dl.api.inference.keras.loaders.preprocessInput(
                    data,
                    tensorShape,
                    inputType = InputType.TF
                )

                return Transpose(axes = intArrayOf(2, 0, 1)).apply(
                    processedData,
                    ImageShape(width = tensorShape[1], height = tensorShape[2], channels = tensorShape[3])
                )
            }
        },

        /** */
        YOLO_v4("models/onnx/objectdetection/yolov4") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                TODO("Not yet implemented")
            }
        },
    }
}
