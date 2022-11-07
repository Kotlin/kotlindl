package org.jetbrains.kotlinx.dl.api.summary

import kotlin.math.max

/**
 * Convenient function to calculate the required width of the column to fit all the values.
 * @param [headerWidth] width of the column header
 * @param [rows] list of rows to calculate the width
 */
public fun calcColumnWidth(headerWidth: Int, rows: List<String>): Int {
    return rows.maxOfOrNull { it.length }?.coerceAtLeast(headerWidth) ?: headerWidth
}

/**
 * Convenient function to calculate the required width of all columns.
 * @param [rows] table with multiple rows each with multiple columns
 * @param [columnNames] list of column names
 *
 * Note: each row must have the same number of columns as the [columnNames] list.
 * Note: each row may consist of multiple lines.
 */
public fun calcColumnWidths(rows: List<TableRow>, columnNames: List<String> = emptyList()): MutableList<Int> {
    if (columnNames.isEmpty()) {
        return listOf(rows.flatMap { it.lines() }.maxOfOrNull { it.size } ?: 0).toMutableList()
    }

    require(rows.all { row ->
        row.lines().all { it.size == columnNames.size }
    }) { "All lines must have the same number of columns" }

    return columnNames.indices.map { columnIndex ->
        val column = rows.map { row -> row.lines() }.flatten().map { it[columnIndex] }
        calcColumnWidth(columnNames[columnIndex].length, column)
    }.toMutableList()
}

/**
 * Format list of strings to a single line with appropriate paddings for each column.
 * @param [columnSeparator] sequence of symbols to separate columns
 * @param [columnsWidths] widths of all columns
 * @param [strs] list of strings to substitute to columns
 */
public fun formatLine(columnSeparator: String, columnsWidths: List<Int>, strs: List<String>): String {
    return columnsWidths
        .mapIndexed { index, columnWidth -> (strs.getOrNull(index) ?: "").padEnd(columnWidth) }
        .joinToString(separator = columnSeparator)
}


/**
 * Pretty print table from multiple sections.
 * Each section consist of multiple rows, each row may consist of multiple lines.
 * Sections may or may not have columns.
 *
 * @see [Section]
 */
public fun formatTable(vararg sections: Section): List<String> {
    require(sections.isNotEmpty()) { "At least one section must be provided" }

    val result = mutableListOf<String>()

    val sectionsWithColumns = sections.filterIsInstance<SectionWithColumns>()

    for (section in sectionsWithColumns) {
        section.alignWith(sectionsWithColumns)
    }

    val simpleSections = sections.filterIsInstance<SimpleSection>()

    for (section in simpleSections) {
        section.alignWith(sectionsWithColumns)
    }

    for (section in sections) {
        result.addAll(section.format())
    }

    return result;
}


/**
 * Base interface for pretty print part of the model summary table.
 */
public sealed interface Section {
    /**
     * Format section to a list of strings.
     */
    public fun format(): List<String> = format(getWidth())

    /**
     * Format section to a list of strings with the specified width.
     */
    public fun format(width: Int): List<String>

    /**
     * Get width of the section.
     */
    public fun getWidth(): Int

    /**
     * Align section with the another section to have all corresponding columns with the same width.
     */
    public fun alignWith(anotherSection: Section)

    /**
     * Align section with the another sections
     * to have all columns equal to the maximum width of the corresponding column of the another sections.
     */
    public fun alignWith(anotherSections: List<Section>)
}


/**
 * Section with columns of the model summary table.
 * Usually it is a header or footer of the table.
 */
public class SimpleSection(
    private val rows: List<TableRow>,
    private val lineSeparatorSymbol: Char,
    private val thickLineSeparatorSymbol: Char
) : Section {
    private var sectionWidth = getWidth()

    override fun format(width: Int): List<String> {
        require(width >= sectionWidth) { "Width must be greater than the sum of column widths" }

        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(width)
        val lineSeparator = lineSeparatorSymbol.toString().repeat(width)

        val result = mutableListOf<String>()

        result.add(thickLineSeparator)

        rows.forEach { row ->
            result.addAll(row.lines().map { it[0] })
        }

        result.add(lineSeparator)

        return result
    }

    override fun alignWith(anotherSection: Section) {
        sectionWidth = max(sectionWidth, anotherSection.getWidth())
    }

    override fun alignWith(anotherSections: List<Section>) {
        sectionWidth = max(sectionWidth, anotherSections.maxOf(Section::getWidth))
    }

    override fun getWidth(): Int {
        return sectionWidth
    }
}

/**
 * Section with columns of the model summary table.
 */
public class SectionWithColumns(
    private val rows: List<TableRow>,
    private val columnNames: List<String>,
    private val columnSeparator: String,
    private val lineSeparatorSymbol: Char,
    private val thickLineSeparatorSymbol: Char
) : Section {
    private var columnsWidths = calcColumnWidths(rows, columnNames)

    override fun format(width: Int): List<String> {
        require(width >= getWidth()) { "Section width must be greater than or equal to ${getWidth()}" }
        updateColumnWidths(width)

        val lineSeparator = lineSeparatorSymbol.toString().repeat(width)
        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(width)

        val result = mutableListOf<String>()

        result.add(formatLine(columnSeparator, columnsWidths, columnNames))

        result.add(thickLineSeparator)

        rows.forEach { row ->
            result.addAll(row.format(columnSeparator, columnsWidths))
            result.add(lineSeparator)
        }

        return result
    }

    override fun alignWith(anotherSection: Section) {
        if (anotherSection == this) return
        if (anotherSection is SimpleSection) return

        anotherSection as SectionWithColumns

        when {
            columnsWidths.size >= anotherSection.columnsWidths.size -> {
                anotherSection.columnsWidths.forEachIndexed { index, columnWidth ->
                    columnsWidths[index] = max(columnWidth, columnsWidths[index])
                }
            }

            else -> {
                columnsWidths.forEachIndexed { index, columnWidth ->
                    columnsWidths[index] = max(columnWidth, anotherSection.columnsWidths[index])
                }

                val extraColumnsWidths = anotherSection.columnsWidths
                    .subList(columnsWidths.size, anotherSection.columnsWidths.size)

                columnsWidths[columnsWidths.lastIndex] +=
                    extraColumnsWidths.sum() + extraColumnsWidths.size * columnSeparator.length
            }
        }
    }

    override fun alignWith(anotherSections: List<Section>) {
        for (section in anotherSections) {
            alignWith(section)
        }
    }

    override fun getWidth(): Int {
        return columnsWidths.sum() + (columnsWidths.size - 1).coerceAtLeast(0) * columnSeparator.length
    }

    private fun updateColumnWidths(newWidth: Int) {
        if (newWidth == getWidth()) return

        columnsWidths = columnsWidths.map { it + (newWidth - getWidth()) / columnsWidths.size }.toMutableList()
    }
}


/**
 * The row of the summary table may consist of multiple lines.
 * This class is used to simplify the handling of formatting of the rows.
 */
public class TableRow(lines: List<List<String>>) {
    private val row: List<List<String>> = lines

    public constructor(line: Collection<String>) : this(listOf(line.toList()))

    public constructor(line: String) : this(listOf(line))

    /**
     * Return the list of lines of the row.
     * Usually it is a list with one element.
     */
    public fun lines(): List<List<String>> {
        return row
    }


    /**
     * Format all lines of the row to fit the specified column widths.
     */
    public fun format(columnSeparator: String, columnsWidths: List<Int>): List<String> {
        return row.map { line -> formatLine(columnSeparator, columnsWidths, line) }
    }
}
