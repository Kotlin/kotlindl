package org.jetbrains.kotlinx.dl.logging.api


/**
 * Wraps color around texts
 *
 * @param color
 */
public inline fun String.wrapColor(color: String): String = "${color}$this${ConsoleColors.RESET}"

/**
 * Console colors
 *
 * Console colors in ansi
 */
public object ConsoleColors {
    // Reset
    public const val RESET: String = "\u001b[0m" // Text Reset

    // Regular Colors
    public const val BLACK : String= "\u001b[0;30m" // BLACK
    public const val RED : String= "\u001b[0;31m" // RED
    public const val GREEN : String= "\u001b[0;32m" // GREEN
    public const val YELLOW : String= "\u001b[0;33m" // YELLOW
    public const val BLUE : String= "\u001b[0;34m" // BLUE
    public const val PURPLE : String= "\u001b[0;35m" // PURPLE
    public const val CYAN : String= "\u001b[0;36m" // CYAN
    public const val WHITE : String= "\u001b[0;37m" // WHITE

    // Bold
    public const val BLACK_BOLD : String= "\u001b[1;30m" // BLACK
    public const val RED_BOLD : String= "\u001b[1;31m" // RED
    public const val GREEN_BOLD : String= "\u001b[1;32m" // GREEN
    public const val YELLOW_BOLD : String= "\u001b[1;33m" // YELLOW
    public const val BLUE_BOLD : String= "\u001b[1;34m" // BLUE
    public const val PURPLE_BOLD : String= "\u001b[1;35m" // PURPLE
    public const val CYAN_BOLD : String= "\u001b[1;36m" // CYAN
    public const val WHITE_BOLD : String= "\u001b[1;37m" // WHITE

    // Underline
    public const val BLACK_UNDERLINED : String= "\u001b[4;30m" // BLACK
    public const val RED_UNDERLINED : String= "\u001b[4;31m" // RED
    public const val GREEN_UNDERLINED : String= "\u001b[4;32m" // GREEN
    public const val YELLOW_UNDERLINED : String= "\u001b[4;33m" // YELLOW
    public const val BLUE_UNDERLINED : String= "\u001b[4;34m" // BLUE
    public const val PURPLE_UNDERLINED : String= "\u001b[4;35m" // PURPLE
    public const val CYAN_UNDERLINED : String= "\u001b[4;36m" // CYAN
    public const val WHITE_UNDERLINED : String= "\u001b[4;37m" // WHITE

    // Background
    public const val BLACK_BACKGROUND : String= "\u001b[40m" // BLACK
    public const val RED_BACKGROUND : String= "\u001b[41m" // RED
    public const val GREEN_BACKGROUND : String= "\u001b[42m" // GREEN
    public const val YELLOW_BACKGROUND : String= "\u001b[43m" // YELLOW
    public const val BLUE_BACKGROUND : String= "\u001b[44m" // BLUE
    public const val PURPLE_BACKGROUND : String= "\u001b[45m" // PURPLE
    public const val CYAN_BACKGROUND : String= "\u001b[46m" // CYAN
    public const val WHITE_BACKGROUND : String= "\u001b[47m" // WHITE

    // High Intensity
    public const val BLACK_BRIGHT : String= "\u001b[0;90m" // BLACK
    public const val RED_BRIGHT : String= "\u001b[0;91m" // RED
    public const val GREEN_BRIGHT : String= "\u001b[0;92m" // GREEN
    public const val YELLOW_BRIGHT : String= "\u001b[0;93m" // YELLOW
    public const val BLUE_BRIGHT : String= "\u001b[0;94m" // BLUE
    public const val PURPLE_BRIGHT : String= "\u001b[0;95m" // PURPLE
    public const val CYAN_BRIGHT : String= "\u001b[0;96m" // CYAN
    public const val WHITE_BRIGHT : String= "\u001b[0;97m" // WHITE

    // Bold High Intensity
    public const val BLACK_BOLD_BRIGHT : String= "\u001b[1;90m" // BLACK
    public const val RED_BOLD_BRIGHT : String= "\u001b[1;91m" // RED
    public const val GREEN_BOLD_BRIGHT : String= "\u001b[1;92m" // GREEN
    public const val YELLOW_BOLD_BRIGHT : String= "\u001b[1;93m" // YELLOW
    public const val BLUE_BOLD_BRIGHT : String= "\u001b[1;94m" // BLUE
    public const val PURPLE_BOLD_BRIGHT : String= "\u001b[1;95m" // PURPLE
    public const val CYAN_BOLD_BRIGHT : String= "\u001b[1;96m" // CYAN
    public const val WHITE_BOLD_BRIGHT : String= "\u001b[1;97m" // WHITE

    // High Intensity backgrounds
    public const val BLACK_BACKGROUND_BRIGHT : String= "\u001b[0;100m" // BLACK
    public const val RED_BACKGROUND_BRIGHT : String= "\u001b[0;101m" // RED
    public const val GREEN_BACKGROUND_BRIGHT : String= "\u001b[0;102m" // GREEN
    public const val YELLOW_BACKGROUND_BRIGHT : String= "\u001b[0;103m" // YELLOW
    public const val BLUE_BACKGROUND_BRIGHT : String= "\u001b[0;104m" // BLUE
    public const val PURPLE_BACKGROUND_BRIGHT : String= "\u001b[0;105m" // PURPLE
    public const val CYAN_BACKGROUND_BRIGHT : String= "\u001b[0;106m" // CYAN
    public const val WHITE_BACKGROUND_BRIGHT : String= "\u001b[0;107m" // WHITE
}