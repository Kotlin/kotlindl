package api

fun <T : Number> getDType(): Class<T> {
    return Float::class.javaObjectType as Class<T>
}