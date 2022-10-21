/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.gradle

private const val ROOT_FOLDER = "models/onnx/mobile"
private const val ORT_EXTENSION = "ort"

/**
 * A list of pretrained models available for download.
 */
enum class ModelType(subfolderName: String, fileName: String, extension: String = ORT_EXTENSION) {
    EfficientNet4Lite("cv", "efficientnet_lite4"),
    MobilenetV1("cv", "mobilenet_v1"),
    MoveNetSinglePoseLighting("poseestimation", "movenet_singlepose_lighting_13"),
    MoveNetSinglePoseThunder("poseestimation", "movenet_thunder"),
    SSDMobileNetV1("objectdetection", "ssd_mobilenet_v1"),
    EfficientDetLite0("objectdetection", "efficientdet_lite0"),
    UltraFace320("facealignment", "ultraface_320"),
    UltraFace640("facealignment", "ultraface_640"),
    Fan2d106("facealignment", "fan_2d_106");

    /**
     * Server path for the model.
     */
    val serverPath = "$ROOT_FOLDER/$subfolderName/$fileName.$extension"
}