package org.jetbrains.kotlinx.dl.logging.core.appenders


import org.jetbrains.kotlinx.dl.logging.api.Appender
import org.jetbrains.kotlinx.dl.logging.api.LogLevel
import org.jetbrains.kotlinx.dl.logging.api.Logger
import org.jetbrains.kotlinx.dl.logging.api.LoggerInternal
import java.io.File
import java.text.SimpleDateFormat
import java.time.LocalDate
import java.time.ZoneId
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * File appender
 *
 * @constructor Creates a new File Appender
 * @param dir the location of logs file directory
 * @property [consoleAppender] used formatter in the console appender
 */
public class FileAppender(private val loggingDir: File = File("logs"), private val consoleAppender: ConsoleAppender) : Appender {
    private val format = SimpleDateFormat("yyyy-MM-dd")

    init {
        loggingDir.mkdir()
    }

    private var date = Date()
    private val blocking = AtomicBoolean(false)

    private var currentFile: File = createNew(date)
    private var currentWriter = currentFile.printWriter()

    /**
     * File watcher thread, change to a new file when in a new day
     */
    @LoggerInternal
    public val fileWatcherThread: Thread = thread {
        while (true) {
            Thread.sleep(1000)
            if (!isSameDay(date, Date())) {
                blocking.set(true)

                currentWriter.flush()
                currentWriter.close()

                date = Date()
                currentFile = createNew(date)
                currentWriter = currentFile.printWriter()

                blocking.set(false)
            }
        }
    }


    override fun logRaw(logger: Logger, level: LogLevel, message: String) {
        while (true) {
            if (blocking.get()) continue
            currentWriter.println(
                consoleAppender.transform(logger, level, message)
                    .replace("\u001B\\[[\\d;]*[^\\d;]".toRegex(), "")
            )
            currentWriter.flush()
            break
        }
    }

    private fun createNew(date: Date): File {
        var index = 0
        var f = File(loggingDir, "log-${format.format(date)}.$index.log")
        while (true) {
            if (f.exists()) {
                index++
                f = File(loggingDir, "log-${format.format(date)}.$index.log")
            } else break
        }
        f.createNewFile()
        return f
    }

    private fun isSameDay(date1: Date, date2: Date): Boolean {
        val localDate1: LocalDate = date1.toInstant()
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
        val localDate2: LocalDate = date2.toInstant()
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
        return localDate1.isEqual(localDate2)
    }
}