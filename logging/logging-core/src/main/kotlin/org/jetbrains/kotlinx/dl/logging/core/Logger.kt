package org.jetbrains.kotlinx.dl.logging.core

import org.jetbrains.kotlinx.dl.logging.api.Appender
import org.jetbrains.kotlinx.dl.logging.api.LogLevel
import org.jetbrains.kotlinx.dl.logging.api.Logger

internal class Logger(override val name: String) : Logger {

    override var level: LogLevel = LogLevel.Info

    override fun log(level: LogLevel, msg: String) {
        val msgLines = msg.split("\\r?\\n".toRegex())
        Appender.appenders.forEach { appender ->
            msgLines.forEach { appender.logRaw(this, level, it) }
        }
    }
}