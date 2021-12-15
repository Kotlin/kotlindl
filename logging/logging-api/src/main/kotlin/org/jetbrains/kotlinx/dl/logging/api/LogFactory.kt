package org.jetbrains.kotlinx.dl.logging.api

import kotlin.reflect.KClass

public lateinit var GlobalLogFactory: LogFactory<out LogFactoryConfig>


public interface LogFactory<C : LogFactoryConfig> {

    public val defaultConfig: C

    public fun newLogger(name: String): Logger

    public fun newLogger(kClass: KClass<*>): Logger

    public fun newLogger(javaClass: Class<*>): Logger

    public fun setup(config: C = defaultConfig) {
        GlobalLogFactory = this
    }

}

public interface LogFactoryConfig