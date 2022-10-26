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
        lineSeparator: Char,
        thickLineSeparator: Char
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

        val inputsColumnHeader = "Inputs"
        val outputsColumnHeader = "Outputs"
        val typeColumnHeader = "Type"

        // Calculate number of columns along with their widths
        val nameColumnWidth =
            allRows.map { it[0] }.calcColumnWidth(maxOf(inputsColumnHeader.length, outputsColumnHeader.length))
        val typeColumnWidth = allRows.map { it[1] }.calcColumnWidth(typeColumnHeader.length)

        val columnsWidths = intArrayOf(nameColumnWidth, typeColumnWidth)

        // Calculate whole table width and prepare strings that will be used as line separators
        val tableWidth = columnsWidths.sum() + (columnsWidths.size - 1).coerceAtLeast(0) * columnSeparator.length
        val lineSeparator = lineSeparator.toString().repeat(tableWidth)
        val thickLineSeparator = thickLineSeparator.toString().repeat(tableWidth)

        val result = mutableListOf<String>()

        result.add(thickLineSeparator)
        result.add("Model type: ONNX")
        result.add(lineSeparator)

        val inputSubtable = formatTable(
            inputRows,
            inputsColumnHeader,
            typeColumnHeader,
            columnSeparator,
            lineSeparator,
            thickLineSeparator,
            columnsWidths
        )

        result.addAll(inputSubtable)
        result.add(thickLineSeparator)

        val outputSubtable = formatTable(
            outputRows,
            outputsColumnHeader,
            typeColumnHeader,
            columnSeparator,
            lineSeparator,
            thickLineSeparator,
            columnsWidths
        )

        result.addAll(outputSubtable)
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


internal fun ValueInfo.summary(): VariableSummary {
    return when (this) {
        is TensorInfo -> summary()
        is MapInfo -> summary()
        is SequenceInfo -> summary()
        else -> throw IllegalStateException("Unknown type of ValueInfo: ${this.javaClass.simpleName}")
    }
}

internal fun TensorInfo.summary() = OnnxTensorVariableSummary(type, TensorShape(shape))

internal fun MapInfo.summary() = OnnxMapVariableSummary(size, keyType, valueType)

internal fun SequenceInfo.summary(): VariableSummary {
    return if (sequenceOfMaps) {
        OnnxMapsSequenceVariableSummary(length, mapInfo.summary())
    } else {
        OnnxSequenceVariableSummary(length, sequenceType)
    }
}
