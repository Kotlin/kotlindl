package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.dataset.Coco
import org.jetbrains.kotlinx.dl.dataset.CocoVersion.V2017
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape


private val SSD_MOBILENET_METADATA = SSDModelMetadata(
    "TFLite_Detection_PostProcess",
    "TFLite_Detection_PostProcess:1",
    "TFLite_Detection_PostProcess:2",
    0, 1
)


public class SSDMobileNetObjectDetectionModel(override val internalModel: OnnxInferenceModel) :
    SSDObjectDetectionModelBase<Bitmap>(SSD_MOBILENET_METADATA),
    InferenceModel by internalModel {

    override val classLabels: Map<Int, String> = Coco.V2017.labels(zeroIndexed = true)

    private var targetRotation = 0f

    override lateinit var preprocessing: Operation<Bitmap, Pair<FloatArray, TensorShape>>
        private set

    public constructor (modelBytes: ByteArray) : this(OnnxInferenceModel(modelBytes)) {
        internalModel.initializeWith(CPU())
        preprocessing = buildPreprocessingPipeline()
    }

    public fun setTargetRotation(targetRotation: Float) {
        if (this.targetRotation == targetRotation) return

        this.targetRotation = targetRotation
        preprocessing = buildPreprocessingPipeline()
    }

    private fun buildPreprocessingPipeline(): Operation<Bitmap, Pair<FloatArray, TensorShape>> {
        return pipeline<Bitmap>()
            .resize {
                outputHeight = inputDimensions[0].toInt()
                outputWidth = inputDimensions[1].toInt()
            }
            .rotate { degrees = targetRotation }
            .toFloatArray { layout = TensorLayout.NHWC }
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): SSDMobileNetObjectDetectionModel {
        return SSDMobileNetObjectDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}
