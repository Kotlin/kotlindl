package org.jetbrains.kotlinx.dl.onnx.summary

import ai.onnxruntime.OnnxJavaType
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.ModelHubModelSummary
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxModelSummary
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxTensorVariableSummary
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class OnnxModelsSummaryTests {
    @Test
    fun formatOfEfficientNetB0InternalModelSummaryTest() {
        assertEquals(
            internalModelExpectedFormat,
            efficientNetB0InternalModelSummary.format()
        )
    }

    @Test
    fun formatOfEfficientNetB0SummaryTest() {
        val header = listOf(
            "=========================================================",
            "EfficientNetB0 model summary",
        )
        assertEquals(
            header + internalModelExpectedFormat,
            efficientNetB0Summary.format()
        )
    }

    private val efficientNetB0InternalModelSummary = OnnxModelSummary(
        inputsSummaries = listOf(
            Pair(
                "input",
                OnnxTensorVariableSummary(
                    dtype = OnnxJavaType.FLOAT,
                    shape = TensorShape(longArrayOf(-1, 224, 224, 3))
                )
            )
        ),
        outputsSummaries = listOf(
            Pair(
                "predictions",
                OnnxTensorVariableSummary(
                    dtype = OnnxJavaType.FLOAT,
                    shape = TensorShape(longArrayOf(-1, 1000))
                )
            )
        )
    )

    private val efficientNetB0Summary = ModelHubModelSummary(
        internalSummary = efficientNetB0InternalModelSummary,
        modelKindDescription = ONNXModels.CV.EfficientNetB0()::class.simpleName
    )

    private val internalModelExpectedFormat = listOf(
        "=========================================================",
        "Model type: ONNX",
        "_________________________________________________________",
        "Inputs      Type                                         ",
        "=========================================================",
        "input       Tensor {dtype=FLOAT, shape [-1, 224, 224, 3]}",
        "_________________________________________________________",
        "Outputs     Type                                         ",
        "=========================================================",
        "predictions Tensor {dtype=FLOAT, shape [-1, 1000]}       ",
        "_________________________________________________________",
    )
}
