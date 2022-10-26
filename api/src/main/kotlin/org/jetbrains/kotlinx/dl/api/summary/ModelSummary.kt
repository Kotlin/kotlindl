package org.jetbrains.kotlinx.dl.api.summary

import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType

/**
 * Common interface for model summary.
 */
public interface ModelSummary {
    /**
     * Formats model summary to array of strings.
     * Rows should form a table with clean and readable structure.
     *
     * @param [columnSeparator] text chunk that will be used as column separator for the output table
     * @param [lineSeparator] character that will be used to produce a string to separate rows of the output table
     * @param [thickLineSeparator] character that will be used to produce a string to separate blocks of the output table
     *
     * @return formatted model summary
     */
    public fun format(
        columnSeparator: String = " ",
        lineSeparator: Char = '_',
        thickLineSeparator: Char = '='
    ): List<String>
}

/**
 * The special kind of model summary for the models that don't implement ModelSummary interface.
 */
public class EmptySummary : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparator: Char,
        thickLineSeparator: Char
    ): List<String> = emptyList()
}

/**
 * The summary for the models from ModelHub.
 * It appends corresponding [ModelType] to the header of the model summary.
 *
 * @property [modelType] type of the model, aka. model architecture. E.g. VGG16, ResNet50, etc.
 * @property [internalSummary] summary of the internal model used for inference
 */
public class ModelHubModelSummary(
    private val internalSummary: ModelSummary,
    private val modelType: ModelType<*, *>?
) : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparator: Char,
        thickLineSeparator: Char
    ): List<String> {
        val table = internalSummary.format(columnSeparator, lineSeparator, thickLineSeparator)

        if (modelType == null) return table

        val modelTypeHeader = "${modelType::class.simpleName} model summary"
        if (table.isEmpty()) return listOf(modelTypeHeader)

        val tableWidth = table.first().length
        val separator = thickLineSeparator.toString().repeat(tableWidth)

        val tableWithHeader = mutableListOf(separator, modelTypeHeader)
        table.forEach(tableWithHeader::add)

        return tableWithHeader
    }
}
