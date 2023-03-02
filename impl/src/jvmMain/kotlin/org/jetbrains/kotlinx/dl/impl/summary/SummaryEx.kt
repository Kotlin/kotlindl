package org.jetbrains.kotlinx.dl.impl.summary

import mu.KLogging
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelWithSummary
import org.slf4j.Logger

private object ModelSummaryLogger : KLogging()

/**
 * Formats and log model summary to logger
 * By defaults prints to [ModelSummaryLogger]
 */
public fun ModelSummary.log(logger: Logger = ModelSummaryLogger.logger): Unit =
    format().forEach(logger::info)

/**
 * Formats and log model summary to logger
 * By defaults prints to [ModelSummaryLogger]
 */
public fun ModelWithSummary.logSummary(logger: Logger = ModelSummaryLogger.logger): Unit =
    summary().log(logger)
