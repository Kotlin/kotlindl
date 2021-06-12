package org.jetbrains.kotlinx.dl.dataset.sound.wav

import java.lang.Exception

/**
 * WavFile-specific exception class.
 */
class WavFileException : Exception {
    /**
     * Create a new WavFile-specific exception.
     */
    constructor() : super() {}

    /**
     * Create a new WavFile-specific exception with a given message.
     *
     * @param message the message
     */
    constructor(message: String?) : super(message) {}

    /**
     * Create a new WavFile-specific exception with a message and throwable exception.
     *
     * @param message the message
     * @param cause the cause
     */
    constructor(message: String?, cause: Throwable?) : super(message, cause) {}

    /**
     * Create a new WavFile-specific exception with a throwable exception.
     *
     * @param cause the cause
     */
    constructor(cause: Throwable?) : super(cause) {}
}