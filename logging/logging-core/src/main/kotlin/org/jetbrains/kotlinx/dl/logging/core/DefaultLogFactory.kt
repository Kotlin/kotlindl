package org.jetbrains.kotlinx.dl.logging.core

import org.jetbrains.kotlinx.dl.logging.api.Appender
import org.jetbrains.kotlinx.dl.logging.api.LogFactory
import org.jetbrains.kotlinx.dl.logging.api.LogFactoryConfig
import org.jetbrains.kotlinx.dl.logging.api.LoggerInternal
import org.jetbrains.kotlinx.dl.logging.core.appenders.ConsoleAppender
import org.jetbrains.kotlinx.dl.logging.core.appenders.FileAppender
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.reflect.KClass
import org.jetbrains.kotlinx.dl.logging.api.Logger as ILogger

/**
 * Default log config
 *
 * @property saveLogs save logs to file
 * @property logDirectory the directory to save logs
 * @constructor Creates a config
 */
public data class DefaultLogConfig(
    val saveLogs: Boolean,
    val logDirectory: File,
) : LogFactoryConfig

/**
 * Default log factory
 *
 *  A lightweight implementation of logging
 */
public object DefaultLogFactory : LogFactory<DefaultLogConfig> {

    override val defaultConfig: DefaultLogConfig = DefaultLogConfig(true, File("logs"))

    @LoggerInternal
    public val maxClassNameLength: AtomicInteger = AtomicInteger(26)

    override fun newLogger(name: String): ILogger = Logger(name)

    override fun newLogger(kClass: KClass<*>): ILogger = Logger(kClass.qualifiedName?.shrinkClassName() ?: "Unknown")

    override fun newLogger(javaClass: Class<*>): ILogger = Logger(javaClass.name.shrinkClassName())

    @OptIn(LoggerInternal::class)
    internal fun String.shrinkClassName(): String {
        val maxLength = maxClassNameLength.get()
        if (length > maxLength) {
            val all = split('.')
            var currentLength = length
            all.forEachIndexed { i, it ->
                currentLength -= it.length + 1
                if (currentLength < maxLength) {
                    return (all.take(i).map { it.first() } + all.takeLast(all.size - i)).joinToString(".")
                }
            }
        }
        return this
    }


    /**
     * Setup Default Implementation
     *
     * Inject Console appender, so we can see output in console
     * Inject File appender(if needed) to save logs
     *
     * @param config the config
     */
    override fun setup(config: DefaultLogConfig) {
        super.setup(config)
        val (saveLogs, logDirectory) = config

        val appenders = Appender.appenders
        val consoleAppender =
            appenders
                .find { it is ConsoleAppender } as ConsoleAppender?
                ?: ConsoleAppender().also { appenders += it }

        if (saveLogs && appenders.filterIsInstance<FileAppender>().isEmpty()) {
            appenders += FileAppender(logDirectory, consoleAppender)
        }
    }

}