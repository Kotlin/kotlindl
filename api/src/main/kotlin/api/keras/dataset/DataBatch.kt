package api.keras.dataset

import java.nio.FloatBuffer

class DataBatch internal constructor(
    private val x: FloatBuffer,
    private val y: FloatBuffer,
    private val numElements: Int
) {
    fun x(): FloatBuffer {
        return x
    }

    fun y(): FloatBuffer {
        return y
    }

    fun shape(elementSize: Int): LongArray {
        return longArrayOf(numElements.toLong(), elementSize.toLong())
    }

    fun size(): Int {
        return numElements
    }
}