/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel

/**
 * Supported models for inference and transfer learning, trained on ImageNet dataset.
 *
 * All weights are imported from the Keras.applications or ONNX.models project and preprocessed with the KotlinDL project.
 *
 * @property [modelName] Name of the model.
 */
public enum class TFModels {
    ;

    public enum class CV(override val modelRelativePath: String) : ModelType {
        /** */
        VGG_16("vgg16") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        VGG_19("vgg19") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_18("resnet18") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_34("resnet34") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_50("resnet50") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_101("resnet101") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_152("resnet151") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        },

        /** */
        ResNet_50_v2("resnet50v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        ResNet_101_v2("resnet101v2"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        ResNet_151_v2("resnet151v2"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        MobileNet("mobilenet"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        MobileNetv2("mobilenetv2"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        Inception("inception"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        Xception("xception"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        DenseNet121("densenet121") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        },

        /** */
        DenseNet169("densenet169"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        },

        /** */
        DenseNet201("densenet201"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        },

        /** */
        NASNetMobile("nasnetmobile"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },

        /** */
        NASNetLarge("nasnetlarge"){
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        },
    }
}

public interface ModelType {
    public val modelRelativePath: String

    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according chosen modelType.
     */
    public fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray
}
