package org.jetbrains.kotlinx.dl.logging.jupyter

import org.jetbrains.kotlinx.dl.logging.api.LogLevel
import org.jetbrains.kotlinx.dl.logging.api.Logger

internal class JupyterLogger(override val name: String) : Logger {

    override var level: LogLevel = LogLevel.Info

    override fun log(level: LogLevel, msg: String) {
        val msgLines = msg.split("\\r?\\n".toRegex())
        msgLines.forEach(::println)
    }
}