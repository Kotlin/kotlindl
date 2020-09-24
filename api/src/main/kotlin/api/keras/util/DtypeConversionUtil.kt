package api.keras.util

/** Returns DType. In existing solution it works with Float only. */
fun getDType(): Class<Float> {
    return Float::class.javaObjectType
}