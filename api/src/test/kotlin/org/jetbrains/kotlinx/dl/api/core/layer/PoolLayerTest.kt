package org.jetbrains.kotlinx.dl.api.core.layer

import org.junit.jupiter.api.Assertions

open class PoolLayerTest : LayerTest() {

    protected fun assertGlobalPoolEquals(
        layer: Layer,
        input: Array<*>,
        expected: Array<FloatArray>,
        inputShape: LongArray,
        expectedShape: LongArray
    ) {
        val actual = createFloatArray2D(shape = expectedShape)
        val output = runLayerInEagerMode(layer, input, inputShape)
        output.copyTo(actual)
        Assertions.assertArrayEquals(expected, actual)
    }
}
