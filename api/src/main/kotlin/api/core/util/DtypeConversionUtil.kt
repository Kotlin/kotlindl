package api.core.util

/** Returns DType. In existing solution it works with Float only. */
internal fun getDType(): Class<Float> {
    return Float::class.javaObjectType
}