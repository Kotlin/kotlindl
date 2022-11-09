package org.jetbrains.kotlinx.dl.api.summary

import kotlin.math.max

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
 * @param [sections] list of sections to print
 * @param [columnSeparator] sequence of symbols to separate columns
 * @param [lineSeparatorSymbol] character that will be used to produce a string to separate rows.
 * @param [thickLineSeparatorSymbol] character that will be used to produce a string to separate sections.
 */
public fun formatTable(
    sections: List<Section>,
    columnSeparator: String = " ",
    lineSeparatorSymbol: Char = '_',
    thickLineSeparatorSymbol: Char = '=',
): List<String> {
    require(sections.isNotEmpty()) { "At least one section must be provided" }

    val sectionsWithColumns = sections.filterIsInstance<SectionWithColumns>()
    val columnsCount = sectionsWithColumns.maxOf { it.columnsCount }
    val columnWidth = List(columnsCount) { column ->
        sectionsWithColumns.maxOfOrNull { section -> section.columnWidth(column) } ?: 0
    }

    val simpleSectionsWidth = sections.filterIsInstance<SimpleSection>().maxOfOrNull(SimpleSection::width) ?: 0
    val sectionsWithColumnsWidth = columnWidth.sum() +
            (columnWidth.size - 1).coerceAtLeast(0) * columnSeparator.length
    val tableWidth = max(simpleSectionsWidth, sectionsWithColumnsWidth)

    val result = mutableListOf<String>()
    for (section in sections) {
        val lines = when (section) {
            is SimpleSection -> section.format(tableWidth, lineSeparatorSymbol, thickLineSeparatorSymbol)
            is SectionWithColumns -> section.format(
                columnWidth, tableWidth,
                columnSeparator, lineSeparatorSymbol, thickLineSeparatorSymbol
            )
        }
        result.addAll(lines)
    }
    return result
}


/**
 * Base interface for pretty print part of the model summary table.
 */
public sealed interface Section

/**
 * Section without columns of the model summary table.
 * Usually it is a header or footer of the table.
 */
public class SimpleSection(private val rows: List<String>) : Section {
    /**
     * Calculate the required width of the section to fit all the lines.
     */
    public val width: Int = rows.maxOf(String::length)

    /**
     * Format section to a list of strings, properly aligned with other sections.
     * @param [tableWidth] required width of the section to be aligned with other sections of the table
     * @param [lineSeparatorSymbol] symbol to use for line separator
     * @param [thickLineSeparatorSymbol] symbol to use for thick line separator
     */
    public fun format(
        tableWidth: Int,
        lineSeparatorSymbol: Char = '_',
        thickLineSeparatorSymbol: Char = '=',
    ): List<String> {
        require(tableWidth >= width) { "Section width must be greater than or equal to $width" }

        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(tableWidth)
        val lineSeparator = lineSeparatorSymbol.toString().repeat(tableWidth)

        val result = mutableListOf<String>()

        result.add(thickLineSeparator)
        result.addAll(rows)
        result.add(lineSeparator)

        return result
    }
}

/**
 * Section with columns of the model summary table.
 * @property [rows] list of rows of the section
 * @property [columnNames] list of names of the columns
 */
public data class SectionWithColumns(
    private val rows: List<TableRow>,
    private val columnNames: List<String>
) : Section {
    /**
     * Number of columns in the section.
     */
    public val columnsCount: Int get() = columnNames.size

    /**
     * Calculate the width of the [column] in the section.
     * It is the maximum width of the column name and all the cells in the column.
     * If section has no column with the given index, returns 0.
     * @param [column] index of the column
     */
    public fun columnWidth(column: Int): Int {
        return rows.maxOf { it.columnWidth(column) }
            .coerceAtLeast(columnNames.getOrNull(column)?.length ?: 0)
    }

    /**
     * Format section to a list of strings, properly aligned with other sections.
     * @param [alignedColumnsWidths] required widths of all columns to be aligned with other sections of the table
     * @param [columnSeparator] sequence of symbols to separate columns
     * @param [lineSeparatorSymbol] symbol to use for line separator
     * @param [thickLineSeparatorSymbol] symbol to use for thick line separator
     */
    public fun format(
        alignedColumnsWidths: List<Int>,
        tableWidth: Int,
        columnSeparator: String = " ",
        lineSeparatorSymbol: Char = '_',
        thickLineSeparatorSymbol: Char = '=',
    ): List<String> {
        val lineSeparator = lineSeparatorSymbol.toString().repeat(tableWidth)
        val thickLineSeparator = thickLineSeparatorSymbol.toString().repeat(tableWidth)

        val result = mutableListOf<String>()

        result.add(formatLine(columnSeparator, alignedColumnsWidths, columnNames))
        result.add(thickLineSeparator)

        rows.forEach { row ->
            result.addAll(row.format(columnSeparator, alignedColumnsWidths))
            result.add(lineSeparator)
        }

        return result
    }
}

/**
 * The row of the summary table may consist of multiple lines.
 * @property cells list of [Cell] of the row.
 */
public class TableRow(private val cells: List<Cell>) {
    public constructor(vararg cells: Cell) : this(cells.toList())

    public constructor(vararg lines: String) : this(lines.toList().map { Cell(it) })

    /**
     * Represent the row as a list of lines
     */
    public val lines: List<List<String>>
        get() {
            val maxLinesInCell = cells.maxOf { it.lines.size }
            val rowLines = IntRange(0, maxLinesInCell - 1).map { idx ->
                cells.map { (cellLines) -> cellLines.getOrNull(idx) ?: " " }
            }
            return rowLines
        }

    /**
     * Calculate the width of the cell in the [column] of the row.
     * If row has no cell with the given index, returns 0.
     * @param [column] index of the column
     */
    public fun columnWidth(column: Int): Int {
        val cell = cells.getOrNull(column) ?: return 0
        return cell.width
    }

    /**
     * Format all lines of the row to fit the specified column widths.
     */
    public fun format(columnSeparator: String, columnsWidths: List<Int>): List<String> {
        return lines.map { line -> formatLine(columnSeparator, columnsWidths, line) }
    }
}

/**
 * This class represents one cell of table row.
 * Cell may consist of multiple lines.
 * This class is used to simplify the handling of formatting of the rows.
 * @property lines list of lines of the cell. Usually it is a list with one element.
 */
public data class Cell(public val lines: List<String>) {
    public constructor(line: String) : this(listOf(line))

    /**
     * Calculate the width of the cell.
     * It is the maximum length of all the lines in the cell.
     * If cell has no lines, returns 0.
     */
    public val width: Int get() = lines.maxOfOrNull { it.length } ?: 0
}
