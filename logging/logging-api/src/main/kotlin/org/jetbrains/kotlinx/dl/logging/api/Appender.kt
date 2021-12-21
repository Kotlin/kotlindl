package org.jetbrains.kotlinx.dl.logging.api


/**
 * Appender interface
 */
public interface Appender {
    /**
     * Log raw
     *
     * @param logger the logger instance when logging
     * @param level the log level
     * @param message the log message
     */
    public fun logRaw(logger: Logger, level: LogLevel, message: String)

    public companion object {
        /**
         * Appenders
         * Global appender lists, simply add appender to this list
         */
        public val appenders: MutableList<Appender> = mutableListOf<Appender>()
    }
}