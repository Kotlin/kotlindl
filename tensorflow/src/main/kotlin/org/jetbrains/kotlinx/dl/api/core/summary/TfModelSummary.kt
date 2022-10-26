package org.jetbrains.kotlinx.dl.api.core.summary

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary

/**
 * The common information about model.
 */
public data class TfModelSummary(
    /** The model type. */
    val type: String,
    /** The model name. */
    val name: String?,
    /** The summary of the all layers included in the model. */
    val layersSummaries: List<LayerSummary>,
    /** The number of trainable parameters. */
    val trainableParamsCount: Long,
    /** The number of frozen parameters. */
    val frozenParamsCount: Long
) : ModelSummary {
    /** The total number of model's parameters. */
    val totalParamsCount: Long = trainableParamsCount + frozenParamsCount
    override fun format(
        columnSeparator: String,
        lineSeparatorSymbol: Char,
        thickLineSeparatorSymbol: Char
    ): List<String> {
        return customFormat(
            columnSeparator = columnSeparator,
            lineSeparatorSymbol = lineSeparatorSymbol,
            thickLineSeparatorSymbol = thickLineSeparatorSymbol
        )
    }

    /**
     * Formats model summary
     * @param [layerNameColumnName] title of the column with layer names
     * @param [outputShapeColumnName] title of the column with layer output shapes
     * @param [paramsCountColumnName] title of the column with layer parameter counts
     * @param [connectedToColumnName] title of the column with layers that are inputs for a layer
     * @param [columnSeparator] text chunk that will be used as column separator for the layer description table
     * @param [lineSeparatorSymbol] character that will be used to produce a string to separate rows of the layer description table
     * @param [thickLineSeparatorSymbol] character that will be used to produce a string to separate general model description,
     * header and body of the table with layers description, and description footer
     * @param [withConnectionsColumn] flag that turns on/off displaying of the column with names of inbound layers
     * @return description of model summary as list of strings, which are suitable for printing or logging
     */
    public fun customFormat(
        layerNameColumnName: String = "Layer (type)                          ",
        outputShapeColumnName: String = "Output Shape             ",
        paramsCountColumnName: String = "Param #      ",
        connectedToColumnName: String = "Connected to               ",
        columnSeparator: String = " ",
        lineSeparatorSymbol: Char = '_',
        thickLineSeparatorSymbol: Char = '=',
        withConnectionsColumn: Boolean = layersSummaries.any { it.inboundLayers.size > 1 }
    ): List<String> {
        // Prepare string data for resulting table
        val table = layersSummaries.map { layer ->
            SummaryRow(
                layerName = "${layer.name}(${layer.type})",
                outputShape = layer.outputShape.toString(),
                paramsCount = layer.paramsCount.toString(),
                inboundLayers = layer.inboundLayers
            )
        }

        // Function to calculate width of column
        fun List<String>.calcColumnWidth(headerWidth: Int) =
            maxOfOrNull { it.length }?.coerceAtLeast(headerWidth) ?: headerWidth

        // Calculate number of columns along with their widths
        val layerNameWidth = table.map { it.layerName }.calcColumnWidth(layerNameColumnName.length)
        val outputShapeWidth = table.map { it.outputShape }.calcColumnWidth(outputShapeColumnName.length)
        val paramsCountWidth = table.map { it.paramsCount }.calcColumnWidth(paramsCountColumnName.length)
        val columnsWidths = if (withConnectionsColumn) {
            val connectedToWidth = table.flatMap { it.inboundLayers }.calcColumnWidth(connectedToColumnName.length)
            intArrayOf(layerNameWidth, outputShapeWidth, paramsCountWidth, connectedToWidth)
        } else {
            intArrayOf(layerNameWidth, outputShapeWidth, paramsCountWidth)
        }

        // Calculate whole table width and prepare strings that will be used as line separators
        val tableWidth = columnsWidths.sum() + (columnsWidths.size - 1).coerceAtLeast(0) * columnSeparator.length
        val lineSeparator = lineSeparatorSymbol.toString().repeat(tableWidth)
        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(tableWidth)

        // Function to format single line of output to look like table row
        fun formatLine(vararg strs: String): String = columnsWidths
            .mapIndexed { index, columnWidth -> (strs.getOrNull(index) ?: "").padEnd(columnWidth) }
            .joinToString(separator = columnSeparator)

        val result = mutableListOf<String>()

        // Header with model type and name
        result.add(thickLineSeparator)
        result.add("Model type: $type")
        name?.let {
            result.add("Model name: $it")
        }
        result.add(lineSeparator)

        // Table header
        result.add(formatLine(layerNameColumnName, outputShapeColumnName, paramsCountColumnName, connectedToColumnName))

        result.add(thickLineSeparator)

        // Table content describing layers
        table.forEach { row ->
            result.add(formatLine(row.layerName, row.outputShape, row.paramsCount, row.inboundLayers.firstOrNull() ?: ""))

            if (withConnectionsColumn)
                row.inboundLayers.drop(1).forEach { result.add(formatLine("", "", "", it)) }

            result.add(lineSeparator)
        }

        result.add(thickLineSeparator)

        // Footer with aggregation info of the model
        result.add("Total trainable params: $trainableParamsCount")
        result.add("Total frozen params: $frozenParamsCount")
        result.add("Total params: $totalParamsCount")

        result.add(thickLineSeparator)

        return result
    }
}

/**
 * The common information about layer.
 */
public data class LayerSummary(
    /** The layer name. */
    val name: String,
    /** The layer type. */
    val type: String,
    /** The output shape of the layer. */
    val outputShape: TensorShape,
    /** The total number of layer's parameters. */
    val paramsCount: Long,
    /** Input layers for the described layer. */
    val inboundLayers: List<String>
)

private data class SummaryRow(
    val layerName: String,
    val outputShape: String,
    val paramsCount: String,
    val inboundLayers: List<String>
)
