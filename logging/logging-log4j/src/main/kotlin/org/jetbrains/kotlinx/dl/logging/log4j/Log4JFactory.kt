package org.jetbrains.kotlinx.dl.logging.log4j

import org.apache.logging.log4j.LogManager
import org.jetbrains.kotlinx.dl.logging.api.LogFactoryConfig
import org.jetbrains.kotlinx.dl.logging.api.LoggerInternal
import java.util.concurrent.atomic.AtomicInteger
import kotlin.reflect.KClass
import org.jetbrains.kotlinx.dl.logging.api.LogFactory as ILogFactory
import org.jetbrains.kotlinx.dl.logging.api.Logger as ILogger

public class Log4JFactoryConfig : LogFactoryConfig

public object Log4JFactory : ILogFactory<Log4JFactoryConfig> {

    override val defaultConfig: Log4JFactoryConfig = Log4JFactoryConfig()

    @LoggerInternal
    public val maxClassNameLength: AtomicInteger = AtomicInteger(26)

    override fun newLogger(name: String): ILogger = LoggerWrapper(LogManager.getLogger(name))

    override fun newLogger(kClass: KClass<*>): ILogger = LoggerWrapper(LogManager.getLogger(kClass.java))

    override fun newLogger(javaClass: Class<*>): ILogger = LoggerWrapper(LogManager.getLogger(javaClass))


}