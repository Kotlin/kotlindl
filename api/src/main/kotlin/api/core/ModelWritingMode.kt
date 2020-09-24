package api.core

/** Model writing mode. */
enum class ModelWritingMode {
    /** Throws an exception if directory exists. */
    FAIL_IF_EXISTS,

    /** Overrides directory if directory exists. */
    OVERRIDE,

    /** Append data to the directory if directory exists. */
    APPEND
}
