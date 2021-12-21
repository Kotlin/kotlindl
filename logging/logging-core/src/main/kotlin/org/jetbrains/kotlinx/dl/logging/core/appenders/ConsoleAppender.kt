package org.jetbrains.kotlinx.dl.logging.core.appenders


import org.jetbrains.kotlinx.dl.logging.api.*
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*

/**
 * Console appender
 *
 * @constructor Creates a new Console Appender
 * @property [format] the format of date
 * @property [transform] the transform method
 */
public class ConsoleAppender(
    private val format: DateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss"),
    public val transform: (Logger, LogLevel, String) -> String = { logger, level, message ->
        "[${
            format.format(Date()).wrapColor(ConsoleColors.PURPLE)
        }] [${logger.name.wrapColor(ConsoleColors.BLUE)}] [${level.formatted}] $message"
    },
) : Appender {

    override fun logRaw(logger: Logger, level: LogLevel, message: String) {
        println(transform(logger, level, message))
    }


}

