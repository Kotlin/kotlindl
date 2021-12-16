package org.jetbrains.kotlinx.dl.logging.api

public interface Logger {
    public val name: String
    public var level: LogLevel


    public fun log(level: LogLevel, msg: String)

    /**
     * Log a warn message.
     *
     * @param msg message to log
     */
    public fun warn(msg: String): Unit = internalLog(LogLevel.Warn, msg)

    /**
     * Log a info message
     *
     * @param msg message to log
     */
    public fun info(msg: String): Unit = internalLog(LogLevel.Info, msg)

    /**
     * Log a debug message
     *
     * @param msg message to log
     */
    public fun debug(msg: String): Unit = internalLog(LogLevel.Debug, msg)

    /**
     * Log a error message
     *
     * @param msg message to log
     */
    public fun error(msg: String): Unit = internalLog(LogLevel.Error, msg)

    /**
     * Log a fetal message
     *
     * @param msg message to log
     */
    public fun fetal(msg: String): Unit = internalLog(LogLevel.Fetal, msg)

    /**
     * Log a useless message
     *
     * @param msg message to log
     */
    public fun useless(msg: String): Unit = internalLog(LogLevel.Useless, msg)


    private fun internalLog(level: LogLevel, msg: String) {
        if (level.order >= this.level.order) {
            log(level, msg)
        }
    }

}

public inline fun Logger.warn(msg: () -> Any?) {
    warn(msg().toString())
}

public inline fun Logger.info(msg: () -> Any?) {
    info(msg().toString())
}

public inline fun Logger.debug(msg: () -> Any?) {
    debug(msg().toString())
}

public inline fun Logger.error(msg: () -> Any?) {
    error(msg().toString())
}

public inline fun Logger.fetal(msg: () -> Any?) {
    fetal(msg().toString())
}

public inline fun Logger.useless(msg: () -> Any?) {
    useless(msg().toString())
}

public inline fun Logger.log(level: LogLevel, msg: () -> Any?) {
    log(level, msg().toString())
}