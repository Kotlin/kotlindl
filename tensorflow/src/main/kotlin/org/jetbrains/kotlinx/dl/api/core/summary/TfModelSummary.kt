package org.jetbrains.kotlinx.dl.api.core.summary

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.*

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
        val headerRows = mutableListOf("Model type: $type")
        name?.let { headerRows.add("Model name: $it") }

        val header = SimpleSection(headerRows)

        val rows = layersSummaries.map { layer ->
            val cells = mutableListOf(
                Cell("${layer.name}(${layer.type})"),
                Cell(layer.outputShape.toString()),
                Cell(layer.paramsCount.toString())
            )
            if (withConnectionsColumn) {
                cells.add(Cell(layer.inboundLayers))
            }

            TableRow(cells)
        }

        val columnNames = if (withConnectionsColumn) {
            listOf(layerNameColumnName, outputShapeColumnName, paramsCountColumnName, connectedToColumnName)
        } else {
            listOf(layerNameColumnName, outputShapeColumnName, paramsCountColumnName)
        }

        val mainSection =
            SectionWithColumns(rows, columnNames)

        val footer = SimpleSection(
            listOf(
                "Total trainable params: $trainableParamsCount",
                "Total frozen params: $frozenParamsCount",
                "Total params: $totalParamsCount"
            )
        )

        return formatTable(
            listOf(header, mainSection, footer),
            columnSeparator,
            lineSeparatorSymbol,
            thickLineSeparatorSymbol
        )
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
