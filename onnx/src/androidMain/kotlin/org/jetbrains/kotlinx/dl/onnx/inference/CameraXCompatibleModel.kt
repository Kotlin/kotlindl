/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

/**
 * Interface represents models which can be used with CameraX API, i.e. support setting of target image rotation.
 */
public interface CameraXCompatibleModel {
    /**
     * Target image rotation.
     * @see androidx.camera.core.ImageInfo
     */
    public var targetRotation: Int
}

/**
 * Convenience function to execute arbitrary code with a preliminary updated target rotation.
 * After the code is executed, the target rotation is restored to its original value.
 *
 * @param rotation target rotation to be set for the duration of the code execution
 * @param function arbitrary code to be executed
 */
public fun <R> CameraXCompatibleModel.doWithRotation(rotation: Int, function: () -> R): R {
    val currentRotation = targetRotation
    targetRotation = rotation
    return function().apply { targetRotation = currentRotation }
}
