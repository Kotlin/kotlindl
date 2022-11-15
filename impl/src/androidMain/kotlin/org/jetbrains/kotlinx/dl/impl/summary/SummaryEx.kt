package org.jetbrains.kotlinx.dl.impl.summary

import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelWithSummary
import java.util.logging.Logger

/**
 * Formats and log model summary to logger
 * By defaults prints to [Logger] with name "KotlinDLLogger"
 */
public fun ModelSummary.log(logger: Logger = Logger.getLogger("KotlinDLLogger")): Unit =
    format().forEach(logger::info)

/**
 * Formats and log model summary to logger
 * By defaults prints to [Logger] with name "KotlinDLLogger"
 */
public fun ModelWithSummary.logSummary(logger: Logger = Logger.getLogger("KotlinDLLogger")): Unit =
    summary().log(logger)
