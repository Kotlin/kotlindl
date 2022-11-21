package org.jetbrains.kotlinx.dl.example.app

import android.content.Context
import android.content.res.Resources
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub

internal class ImageAnalyzer(
    context: Context,
    private val resources: Resources,
    private val uiUpdateCallBack: (AnalysisResult?) -> Unit,
    initialPipelineIndex: Int = 0
) {
    private val hub = ONNXModelHub(context)

    val pipelinesList = Pipelines.values().sortedWith(Comparator { o1, o2 ->
        if (o1.task != o2.task) return@Comparator o1.task.ordinal - o2.task.ordinal
        o1.ordinal - o2.ordinal
    })
    private val pipelines = pipelinesList.map { it.createPipeline(hub, resources) }

    @Volatile
    var currentPipelineIndex: Int = initialPipelineIndex
        private set
    private val currentPipeline: InferencePipeline? get() = pipelines.getOrNull(currentPipelineIndex)

    fun analyze(image: ImageProxy, isImageFlipped: Boolean) {
        val pipeline = currentPipeline
        if (pipeline == null) {
            uiUpdateCallBack(null)
            return
        }

        val start = SystemClock.uptimeMillis()
        val result = pipeline.analyze(image, confidenceThreshold)
        val end = SystemClock.uptimeMillis()

        val rotationDegrees = image.imageInfo.rotationDegrees
        image.close()

        if (result == null || result.confidence < confidenceThreshold) {
            uiUpdateCallBack(AnalysisResult.Empty(end - start))
        } else {
            uiUpdateCallBack(
                AnalysisResult.WithPrediction(
                    result, end - start,
                    ImageMetadata(image.width, image.height, isImageFlipped, rotationDegrees)
                )
            )
        }
    }

    fun setPipeline(index: Int) {
        currentPipelineIndex = index
    }

    fun clear() {
        currentPipelineIndex = -1
    }

    fun close() {
        clear()
        pipelines.forEach(InferencePipeline::close)
    }

    companion object {
        private const val confidenceThreshold = 0.5f
    }
}

sealed class AnalysisResult(val processTimeMs: Long) {
    class Empty(processTimeMs: Long) : AnalysisResult(processTimeMs)
    class WithPrediction(
        val prediction: Prediction,
        processTimeMs: Long,
        val metadata: ImageMetadata
    ) : AnalysisResult(processTimeMs)
}

interface Prediction {
    val shapes: List<FlatShape<*>>
    val confidence: Float
    fun getText(context: Context): String
}

data class ImageMetadata(
    val width: Int,
    val height: Int,
    val isImageFlipped: Boolean
) {

    constructor(width: Int, height: Int, isImageFlipped: Boolean, rotationDegrees: Int)
            : this(
        if (areDimensionSwitched(rotationDegrees)) height else width,
        if (areDimensionSwitched(rotationDegrees)) width else height,
        isImageFlipped
    )

    companion object {
        private fun areDimensionSwitched(rotationDegrees: Int): Boolean {
            return rotationDegrees == 90 || rotationDegrees == 270
        }
    }
}