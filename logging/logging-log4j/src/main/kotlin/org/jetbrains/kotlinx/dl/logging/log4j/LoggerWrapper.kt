package org.jetbrains.kotlinx.dl.logging.log4j

import org.apache.logging.log4j.Level
import org.jetbrains.kotlinx.dl.logging.api.LogLevel
import org.jetbrains.kotlinx.dl.logging.api.Logger
import org.apache.logging.log4j.Logger as Log4jLogger

private val internal2log4j = mapOf(
    LogLevel.Fetal to Level.FATAL,
    LogLevel.Error to Level.ERROR,
    LogLevel.Warn to Level.WARN,
    LogLevel.Info to Level.INFO,
    LogLevel.Debug to Level.DEBUG,
    LogLevel.Useless to Level.ALL
)

private val log4j2internal = internal2log4j.entries.associate { (k, v) -> v to k }


@JvmInline
internal value class LoggerWrapper(internal val internalLogger: Log4jLogger) : Logger {

    override val name: String get() = internalLogger.name
    override var level: LogLevel
        get() = log4j2internal[internalLogger.level]!!
        set(value) {
            // can't change level in log4j
        }

    override fun log(level: LogLevel, msg: String) {
        internalLogger.log(internal2log4j[level], msg)
    }
}

public fun Logger.toLog4J(): Log4jLogger = (this as LoggerWrapper).internalLogger