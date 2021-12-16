package org.jetbrains.kotlinx.dl.api.core.dsl

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Layer

@DslMarker
@Retention(AnnotationRetention.SOURCE)
internal annotation class BuilderDsl

@DslMarker
@Retention(AnnotationRetention.SOURCE)
internal annotation class EntryDsl

public class GraphTrainableModelDsl<T : GraphTrainableModel>(private val initializer: (Array<Layer>) -> T) {

    private var handler: T.() -> Unit = {}
    private var layerProvider: LayerBuilder.() -> Unit = {}
    private var use: ((T) -> Unit)? = null

    @BuilderDsl
    public fun model(builder: T.() -> Unit) {
        this.handler = builder
    }

    @BuilderDsl
    public fun layers(builder: LayerBuilder.() -> Unit) {
        this.layerProvider = builder
    }

    @BuilderDsl
    public fun use(builder: T.() -> Unit) {
        use = builder
    }

    public fun build(): T = initializer.invoke(LayerBuilder().apply(layerProvider).toArray()).apply(handler).apply {
        use?.let { use(it) }
    }
}

@JvmInline
public value class LayerBuilder(private val list: MutableList<Layer> = mutableListOf()) {

    public operator fun Layer.unaryPlus(): Layer {
        list.add(this)
        return this
    }

    public fun toArray(): Array<Layer> = list.toTypedArray()

    public inline operator fun <T : Layer> T.invoke(builder: T.() -> Unit) {
        this.apply(builder)
    }
}


@EntryDsl
public fun sequential(builder: GraphTrainableModelDsl<Sequential>.() -> Unit): Sequential {
    return GraphTrainableModelDsl(Sequential.Companion::of).apply(builder).build()
}

@EntryDsl
public fun functional(builder: GraphTrainableModelDsl<Functional>.() -> Unit): Functional {
    return GraphTrainableModelDsl(Functional.Companion::of).apply(builder).build()
}

@EntryDsl
public operator fun <T : GraphTrainableModel> ((Array<Layer>) -> T).invoke(builder: GraphTrainableModelDsl<T>.() -> Unit): T {
    return GraphTrainableModelDsl(this).apply(builder).build()
}
