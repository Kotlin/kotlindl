package org.jetbrains.kotlinx.dl.logging.jupyter

import org.jetbrains.kotlinx.dl.logging.api.LogFactoryConfig
import kotlin.reflect.KClass
import org.jetbrains.kotlinx.dl.logging.api.LogFactory as ILogFactory
import org.jetbrains.kotlinx.dl.logging.api.Logger as ILogger

public class JupyterLoggerConfig : LogFactoryConfig

/**
 * Jupyter factory
 *
 */
public object JupyterFactory : ILogFactory<JupyterLoggerConfig> {


    override val defaultConfig: JupyterLoggerConfig = JupyterLoggerConfig()

    public var currentConfig: JupyterLoggerConfig = defaultConfig

    override fun newLogger(name: String): ILogger = JupyterLogger(name)

    override fun newLogger(kClass: KClass<*>): ILogger = JupyterLogger(kClass.qualifiedName ?: "Unknown")

    override fun newLogger(javaClass: Class<*>): ILogger = JupyterLogger(javaClass.name)


    override fun setup(config: JupyterLoggerConfig) {
        super.setup(config)
        currentConfig = config
    }

}