package tensorflow.training.util

import java.nio.FloatBuffer

class ImageBatch internal constructor(
    private val images: FloatBuffer,
    private val labels: FloatBuffer,
    private val numElements: Int
) {
    fun images(): FloatBuffer {
        return images
    }

    fun labels(): FloatBuffer {
        return labels
    }

    fun shape(elementSize: Int): LongArray {
        return longArrayOf(numElements.toLong(), elementSize.toLong())
    }

    fun size(): Int {
        return numElements
    }
}