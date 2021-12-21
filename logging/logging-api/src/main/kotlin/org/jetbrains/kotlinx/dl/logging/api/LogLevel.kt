package org.jetbrains.kotlinx.dl.logging.api


/**
 * Log level
 *
 * @property [order] the order (priority)
 * @constructor
 *
 * @param colorString
 */
public enum class LogLevel(public val order: Int, colorString: String = ConsoleColors.BLACK_BRIGHT) {
    /**
     * Fetal level
     */
    Fetal(10000, ConsoleColors.RED),

    /**
     * Error level
     */
    Error(8000, ConsoleColors.RED),

    /**
     * Warning level
     */
    Warn(6000, ConsoleColors.GREEN),

    /**
     * Warning level
     */
    Info(4000, ConsoleColors.GREEN),

    /**
     * Debug level
     */
    Debug(2000, ConsoleColors.BLUE),

    /**
     * Useless level
     */
    Useless(0);


    /**
     * Formatted with name and color
     */
    public val formatted: String = name.wrapColor(colorString)

}