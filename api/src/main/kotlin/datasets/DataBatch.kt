package datasets

import java.nio.FloatBuffer

/**
 * This class represents the batch of data in [Dataset].
 * @param [x] Data observations.
 * @param [y] Labels.
 * @param [size] Number of rows in batch.
 */
public data class DataBatch internal constructor(val x: FloatBuffer, val y: FloatBuffer, val size: Int) {
    /** */
    public fun shape(elementSize: Int): List<Long> = listOf(size.toLong(), elementSize.toLong())
}
