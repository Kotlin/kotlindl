package org.jetbrains.kotlinx.dl.api.inference.onnx

/**
 * Interface represents models which can be used with CameraX API, i.e. support setting of target image rotation.
 */
public interface CameraXCompatibleModel {
    /**
     * Target image rotation.
     * @see [ImageInfo](https://developer.android.com/reference/androidx/camera/core/ImageInfo)
     */
    public var targetRotation: Float
}
