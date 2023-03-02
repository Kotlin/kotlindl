package org.jetbrains.kotlinx.dl.api.summary

/**
 * Common interface for model summary.
 */
public interface ModelSummary {
    /**
     * Formats model summary to array of strings.
     * Rows should form a table with clean and readable structure.
     *
     * @param [columnSeparator] text chunk that will be used as column separator for the output table
     * @param [lineSeparatorSymbol] character that will be used to produce a string to separate rows of the output table
     * @param [thickLineSeparatorSymbol] character that will be used to produce a string to separate blocks of the output table
     *
     * @return formatted model summary
     */
    public fun format(
        columnSeparator: String = " ",
        lineSeparatorSymbol: Char = '_',
        thickLineSeparatorSymbol: Char = '='
    ): List<String>
}

/**
 * The special kind of model summary for the models that don't implement ModelSummary interface.
 */
public class EmptySummary : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparatorSymbol: Char,
        thickLineSeparatorSymbol: Char
    ): List<String> = emptyList()
}

/**
 * The summary for the models from ModelHub.
 * It appends corresponding [modelKindDescription] to the header of the model summary.
 *
 * @property [modelKindDescription] type of the model, aka. model architecture. E.g. VGG16, ResNet50, etc.
 * @property [internalSummary] summary of the internal model used for inference
 */
public class ModelHubModelSummary(
    private val internalSummary: ModelSummary,
    private val modelKindDescription: String? = null
) : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparatorSymbol: Char,
        thickLineSeparatorSymbol: Char
    ): List<String> {
        val table = internalSummary.format(columnSeparator, lineSeparatorSymbol, thickLineSeparatorSymbol)

        if (modelKindDescription == null) return table

        val modelTypeHeader = "$modelKindDescription model summary"
        if (table.isEmpty()) return listOf(modelTypeHeader)

        val tableWidth = table.first().length
        val separator = thickLineSeparatorSymbol.toString().repeat(tableWidth)

        val tableWithHeader = mutableListOf(separator, modelTypeHeader)
        tableWithHeader.addAll(table)

        return tableWithHeader
    }
}
