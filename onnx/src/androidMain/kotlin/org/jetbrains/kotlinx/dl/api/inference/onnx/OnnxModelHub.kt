package org.jetbrains.kotlinx.dl.api.inference.onnx

import android.content.Context
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider

/**
 * This class provides methods for loading ONNX model from android application resources.
 */
public class ONNXModelHub(private val context: Context) : ModelHub() {
    /**
     * Loads ONNX model from android resources.
     * By default, the model is initialized with [ExecutionProvider.CPU] execution provider.
     *
     * @param [modelType] model type from [ONNXModels]
     * @param [executionProviders] execution providers for model initialization.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: OnnxModelType<T, U>,
        vararg executionProviders: ExecutionProvider = arrayOf(ExecutionProvider.CPU())
    ): T {
        val modelResourceId = context.resources.getIdentifier(modelType.modelRelativePath, "raw", context.packageName)
        val inferenceModel = OnnxInferenceModel(context.resources.openRawResource(modelResourceId).readBytes())
        modelType.inputShape?.let { shape -> inferenceModel.reshape(*shape) }
        inferenceModel.initializeWith(*executionProviders)
        return inferenceModel as T
    }


    override fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode /* unused */,
    ): T {
        return loadModel(modelType as OnnxModelType<T, U>)
    }
}
