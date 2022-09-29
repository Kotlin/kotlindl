package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Coco
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * Special model class for detection objects on images with built-in preprocessing and post-processing.
 * Suitable for models with SSD like output decoding.
 *
 * It internally uses [ONNXModels.ObjectDetection.EfficientDetLite0] or other SSDLike models trained on the COCO dataset.
 *
 * @param [internalModel] model used to make predictions
 *
 * @since 0.5
 */
public class SSDLikeModel(override val internalModel: OnnxInferenceModel, metadata: SSDLikeModelMetadata) :
    SSDLikeModelBase<Bitmap>(metadata), CameraXCompatibleModel, InferenceModel by internalModel {

    override val classLabels: Map<Int, String> = Coco.V2017.labels(zeroIndexed = true)

    override var targetRotation: Float = 0f

    override val preprocessing: Operation<Bitmap, Pair<FloatArray, TensorShape>>
        get() = pipeline<Bitmap>()
            .resize {
                outputHeight = internalModel.inputDimensions[0].toInt()
                outputWidth = internalModel.inputDimensions[1].toInt()
            }
            .rotate { degrees = targetRotation }
            .toFloatArray { layout = TensorLayout.NHWC }

    override fun close() {
        internalModel.close()
    }
}
