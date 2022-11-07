package org.jetbrains.kotlinx.dl.onnx.inference

import ai.onnxruntime.MapInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.SequenceInfo
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.ValueInfo
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.*
import java.lang.IllegalStateException

/**
 * Summary of the OnnxInferenceModel.
 * Contains information about input and output variables of the model.
 *
 * @param [inputsSummaries] list of summaries of the input variables along with their names
 * @param [outputsSummaries] list of summaries of the output variables along with their names
 */
public data class OnnxModelSummary(
    private val inputsSummaries: List<Pair<String, VariableSummary>>,
    private val outputsSummaries: List<Pair<String, VariableSummary>>,
) : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparatorSymbol: Char,
        thickLineSeparatorSymbol: Char
    ): List<String> {
        return customFormat(columnSeparator, lineSeparatorSymbol, thickLineSeparatorSymbol)
    }

    /**
     * Format function with customizable column names.
     * @param [inputsColumnHeader] title of the column with input variables
     * @param [outputsColumnHeader] title of the column with output variables
     * @param [typeColumnHeader] title of the column with variable types
     */
    public fun customFormat(
        columnSeparator: String,
        lineSeparatorSymbol: Char,
        thickLineSeparatorSymbol: Char,
        inputsColumnHeader: String = "Inputs",
        outputsColumnHeader: String = "Outputs",
        typeColumnHeader: String = "Type"
    ): List<String> {
        val inputRows = inputsSummaries.map { (name, summary) ->
            TableRow(listOf(name, summary.toSummaryRow()))
        }

        val outputRows = outputsSummaries.map { (name, summary) ->
            TableRow(listOf(name, summary.toSummaryRow()))
        }

        val header = SimpleSection(
            listOf(TableRow("Model type: ONNX")),
            lineSeparatorSymbol,
            thickLineSeparatorSymbol
        )

        val inputsSection = SectionWithColumns(
            inputRows,
            listOf(inputsColumnHeader, typeColumnHeader),
            columnSeparator, lineSeparatorSymbol, thickLineSeparatorSymbol
        )

        val outputsSection = SectionWithColumns(
            outputRows,
            listOf(outputsColumnHeader, typeColumnHeader),
            columnSeparator, lineSeparatorSymbol, thickLineSeparatorSymbol
        )

        return formatTable(header, inputsSection, outputsSection)
    }
}

/**
 * Interface to produce text description of the ONNX variable.
 */
public interface VariableSummary {
    /**
     * Returns text description of the variable.
     */
    public fun toSummaryRow(): String
}

internal data class OnnxTensorVariableSummary(
    val dtype: OnnxJavaType,
    val shape: TensorShape
) : VariableSummary {
    override fun toSummaryRow() = "Tensor {dtype=${dtype.name}, shape [${shape.dims().joinToString()}]}"
}

internal data class OnnxMapVariableSummary(
    val size: Int,
    val keyType: OnnxJavaType,
    val valueType: OnnxJavaType
) : VariableSummary {
    override fun toSummaryRow() = "Map {${keyType.name}: ${valueType.name}, size=$size}"
}

internal data class OnnxMapsSequenceVariableSummary(
    val length: Int,
    val mapSummary: OnnxMapVariableSummary,
) : VariableSummary {
    override fun toSummaryRow() = "Sequence of Maps {${mapSummary.toSummaryRow()}, length=$length}"
}

internal data class OnnxSequenceVariableSummary(
    val length: Int,
    val dtype: OnnxJavaType,
) : VariableSummary {
    override fun toSummaryRow() = "Seqeunce {dtype=${dtype.name}, length=$length}"
}

internal fun ValueInfo.summary() = when {
    this is TensorInfo -> OnnxTensorVariableSummary(type, TensorShape(shape))
    this is MapInfo -> OnnxMapVariableSummary(size, keyType, valueType)
    this is SequenceInfo && sequenceOfMaps -> OnnxMapsSequenceVariableSummary(
        length,
        OnnxMapVariableSummary(mapInfo.size, mapInfo.keyType, mapInfo.valueType)
    )

    this is SequenceInfo -> OnnxSequenceVariableSummary(length, sequenceType)
    else -> throw IllegalStateException("Unknown type of ValueInfo: ${this.javaClass.simpleName}")
}
