package org.jetbrains.kotlinx.dl.api.summary

import java.io.PrintStream

/**
 * Formats and prints model summary to output stream
 * By defaults prints to console
 */
public fun ModelSummary.print(out: PrintStream = System.out): Unit =
    format().forEach(out::println)

/**
 * Formats and prints model summary to output stream
 * By defaults prints to console
 */
public fun ModelWithSummary.printSummary(out: PrintStream = System.out): Unit =
    summary().print(out)
