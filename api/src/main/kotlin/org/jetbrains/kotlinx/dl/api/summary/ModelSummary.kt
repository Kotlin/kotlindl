package org.jetbrains.kotlinx.dl.api.summary

import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType

public interface ModelSummary {
    public fun format(
        columnSeparator: String = " ",
        lineSeparator: Char = '_',
        thickLineSeparator: Char = '='
    ): List<String>
}

public class EmptySummary : ModelSummary {
    override fun format(
        columnSeparator: String,
        lineSeparator: Char,
        thickLineSeparator: Char
    ): List<String> = emptyList()
}

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
