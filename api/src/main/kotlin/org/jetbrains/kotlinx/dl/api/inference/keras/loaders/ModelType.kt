/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

public enum class ModelType(public val modelName: String) {
    VGG_16("vgg16"),
    VGG_19("vgg19"),
    ResNet_50("resnet50"),
    ResNet_101("resnet101"),
    ResNet_152("resnet151"),
    ResNet_50_v2("resnet50v2"),
    ResNet_101_v2("resnet101v2"),
    ResNet_151_v2("resnet151v2"),

    /**
     * Loads MobileNet model.
     *
     * NOTE: All layers will be frozen!
     */
    MobileNet("mobilenet"),

    /**
     * Loads MobileNetV2 model.
     *
     * NOTE: All layers will be frozen!
     */
    MobileNetv2("mobilenetv2"),
}
