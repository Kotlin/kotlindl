package org.jetbrains.kotlinx.dl.logging.api

import kotlin.reflect.KClass

public lateinit var GlobalLogFactory: LogFactory<out LogFactoryConfig>

/**
 * Log factory interface
 *
 * @param C the config class
 */
public interface LogFactory<C : LogFactoryConfig> {

    /**
     * Default config
     */
    public val defaultConfig: C

    /**
     * New logger
     *
     * creates a new logger with [name]
     *
     * @param name the name of the logger
     * @return a logger implementation
     */
    public fun newLogger(name: String): Logger

    /**
     * New logger
     *
     * creates a new logger with [kClass]
     *
     * @param kClass the Kotlin Class to log with
     * @return a logger implementation
     */
    public fun newLogger(kClass: KClass<*>): Logger

    /**
     * New logger
     *
     * creates a new logger with [javaClass]
     *
     * @param javaClass the Java Class to log with
     * @return a logger implementation
     */
    public fun newLogger(javaClass: Class<*>): Logger

    /**
     * Setup
     *
     * @param config the config for this factory, defaults to [defaultConfig]
     */
    public fun setup(config: C = defaultConfig) {
        GlobalLogFactory = this
    }

}

/**
 * Log factory config interface
 *
 */
public interface LogFactoryConfig