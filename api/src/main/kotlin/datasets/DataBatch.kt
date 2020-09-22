package datasets

import java.nio.FloatBuffer

public data class DataBatch internal constructor(val x: FloatBuffer, val y: FloatBuffer, val size: Int) {
    public fun shape(elementSize: Int): List<Long> = listOf(size.toLong(), elementSize.toLong())
}
