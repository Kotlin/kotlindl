package examples

import org.jetbrains.kotlinx.dl.logging.api.GlobalLogFactory
import org.jetbrains.kotlinx.dl.logging.core.DefaultLogFactory
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test


object Basic {
    @Test
    @BeforeAll
    @JvmStatic
    fun initLogger() {
        GlobalLogFactory = DefaultLogFactory.also {
            it.setup()
        }
    }
}