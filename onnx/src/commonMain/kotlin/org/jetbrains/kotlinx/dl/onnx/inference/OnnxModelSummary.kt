package org.jetbrains.kotlinx.dl.onnx.inference

import ai.onnxruntime.MapInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.SequenceInfo
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.ValueInfo
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
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
            listOf(name, summary.toSummaryRow())
        }

        val outputRows = outputsSummaries.map { (name, summary) ->
            listOf(name, summary.toSummaryRow())
        }

        val allRows = inputRows + outputRows

        fun List<String>.calcColumnWidth(headerWidth: Int) =
            maxOfOrNull { it.length }?.coerceAtLeast(headerWidth) ?: headerWidth

        // Calculate number of columns along with their widths
        val nameColumnWidth =
            allRows.map { it[0] }.calcColumnWidth(maxOf(inputsColumnHeader.length, outputsColumnHeader.length))
        val typeColumnWidth = allRows.map { it[1] }.calcColumnWidth(typeColumnHeader.length)

        val columnsWidths = intArrayOf(nameColumnWidth, typeColumnWidth)

        // Calculate whole table width and prepare strings that will be used as line separators
        val tableWidth = columnsWidths.sum() + (columnsWidths.size - 1).coerceAtLeast(0) * columnSeparator.length
        val lineSeparator = lineSeparatorSymbol.toString().repeat(tableWidth)
        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(tableWidth)

        val result = mutableListOf<String>()

        result.add(thickLineSeparator)
        result.add("Model type: ONNX")
        result.add(lineSeparator)
        result.add(thickLineSeparator)

        val inputSubTable = formatTable(
            inputRows,
            inputsColumnHeader,
            typeColumnHeader,
            columnSeparator,
            lineSeparator,
            thickLineSeparator,
            columnsWidths
        )

        result.addAll(inputSubTable)
        result.add(thickLineSeparator)

        val outputSubTable = formatTable(
            outputRows,
            outputsColumnHeader,
            typeColumnHeader,
            columnSeparator,
            lineSeparator,
            thickLineSeparator,
            columnsWidths
        )

        result.addAll(outputSubTable)
        result.add(thickLineSeparator)

        return result
    }

    private fun formatTable(
        table: List<List<String>>,
        nameColumnHeader: String,
        typeColumnHeader: String,
        columnSeparator: String,
        lineSeparator: String,
        thickLineSeparator: String,
        columnsWidths: IntArray
    ): List<String> {
        fun formatLine(vararg strs: String): String = columnsWidths
            .mapIndexed { index, columnWidth -> (strs.getOrNull(index) ?: "").padEnd(columnWidth) }
            .joinToString(separator = columnSeparator)

        val result = mutableListOf<String>()

        result.add(formatLine(nameColumnHeader, typeColumnHeader))

        result.add(thickLineSeparator)

        table.forEach { row ->
            result.add(formatLine(row[0], row[1]))
            result.add(lineSeparator)
        }

        return result
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
