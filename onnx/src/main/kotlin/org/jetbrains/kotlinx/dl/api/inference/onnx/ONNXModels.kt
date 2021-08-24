/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx;

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType

public enum class ONNXModels {
        ;
        public enum class CV(override val modelRelativePath: String) : ModelType {
            /** */
            ResNet_18_v1("models/onnx/cv/resnet/resnet18-v1"),

            /** */
            ResNet_34_v1("models/onnx/cv/resnet/resnet34-v1"),

            /** */
            ResNet_50_v1("models/onnx/cv/resnet/resnet50-v1"),

            /** */
            ResNet_101_v1("models/onnx/cv/resnet/resnet101-v1"),

            /** */
            ResNet_152_v1("models/onnx/cv/resnet/resnet151-v1"),

            /** */
            ResNet_18_v2("models/onnx/cv/resnet/resnet18-v2"),

            /** */
            ResNet_34_v2("models/onnx/cv/resnet/resnet34-v2"),

            /** */
            ResNet_50_v2("models/onnx/cv/resnet/resnet50-v2"),

            /** */
            ResNet_101_v2("models/onnx/cv/resnet/resnet101-v2"),

            /** */
            ResNet_151_v2("models/onnx/cv/resnet/resnet151-v2"),

            /** */
            DenseNet_121("models/onnx/cv/densenet/densenet121"),

            /** */
            EfficientNet_4_Lite("models/onnx/cv/efficientnet/efficientnet-lite4"),

            /** */
            ResNet_50_v1_custom("models/onnx/cv/custom/resnet50"),

            /** */
            ResNet_50_v1_no_top_custom("models/onnx/cv/custom/resnet50notop"),

            /** */
            Lenet_mnist("models/onnx/cv/custom/mnist")
        }
        ;

        public enum class ObjectDetection(override val modelRelativePath: String) : ModelType {
            /** */
            SSD("models/onnx/objectdetection/ssd"),

            /** */
            YOLO_v4("models/onnx/objectdetection/yolov4"),
        }
    }
