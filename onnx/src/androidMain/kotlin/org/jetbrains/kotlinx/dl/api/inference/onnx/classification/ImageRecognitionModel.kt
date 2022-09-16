package org.jetbrains.kotlinx.dl.api.inference.onnx.classification

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModelBase
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.onnx.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ExecutionProviderCompatible
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.dataset.imagenetLabels
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * The light-weight API for Classification task with one of the Model Hub models.
 */
public open class ImageRecognitionModel(
    internalModel: OnnxInferenceModel,
    private val modelType: ModelType<out InferenceModel, out InferenceModel>,
    override val classLabels: Map<Int, String> = Imagenet.V1k.labels()
) : ImageRecognitionModelBase<Bitmap>(internalModel), ExecutionProviderCompatible, CameraXCompatibleModel {
    override var targetRotation: Float = 0f

    override val preprocessing: Operation<Bitmap, Pair<FloatArray, TensorShape>>
        get() {
            val (width, height) = if (modelType.channelsFirst)
                Pair(internalModel.inputDimensions[1], internalModel.inputDimensions[2])
            else
                Pair(internalModel.inputDimensions[0], internalModel.inputDimensions[1])

            return pipeline<Bitmap>()
                .resize {
                    outputHeight = height.toInt()
                    outputWidth = width.toInt()
                }
                .rotate { degrees = targetRotation }
                .toFloatArray { layout = if (modelType.channelsFirst) TensorLayout.NCHW else TensorLayout.NHWC }
                .call(modelType.preprocessor)
        }

    override fun initializeWith(vararg executionProviders: ExecutionProvider) {
        (internalModel as OnnxInferenceModel).initializeWith(*executionProviders)
    }
}
