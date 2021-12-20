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

public class GraphTrainableModelBuilder<T : GraphTrainableModel>(private val createModelFunction: (Array<Layer>) -> T) {

    private var modelBuilderBlock: T.() -> Unit = {}
    private var layerListBuilderBlock: LayerListBuilder.() -> Unit = {}
    private var useBlock: ((T) -> Unit)? = null

    @BuilderDsl
    public fun model(block: T.() -> Unit) {
        modelBuilderBlock = block
    }

    @BuilderDsl
    public fun layers(block: LayerListBuilder.() -> Unit) {
        layerListBuilderBlock = block
    }

    @BuilderDsl
    public fun use(block: T.() -> Unit) {
        useBlock = block
    }

    public fun build(): T = createModelFunction(LayerListBuilder().apply(layerListBuilderBlock).toArray())
        .apply {
            modelBuilderBlock()
            useBlock?.let { use(it) }
        }

}

/**
 * Layer list builder
 *
 * @property [layers] a mutable list to store layers
 * @constructor Create empty Layer list builder
 */
@JvmInline
public value class LayerListBuilder(private val layers: MutableList<Layer> = mutableListOf()) {

    /**
     * Unary plus
     *
     * Add a layer to the list and return itself.
     *
     * We can do this
     *
     *     +Input(128, 128)
     *
     * instead of
     *
     *     layers.add(Input(128, 128))
     *
     * @return the layer
     */
    public operator fun Layer.unaryPlus(): Layer {
        layers.add(this)
        return this
    }

    /**
     * To array
     *
     * @return array of layer
     */
    public fun toArray(): Array<Layer> = layers.toTypedArray()
}


@EntryDsl
public fun sequential(builder: GraphTrainableModelBuilder<Sequential>.() -> Unit): Sequential {
    return GraphTrainableModelBuilder(Sequential.Companion::of).apply(builder).build()
}

@EntryDsl
public fun functional(builder: GraphTrainableModelBuilder<Functional>.() -> Unit): Functional {
    return GraphTrainableModelBuilder(Functional.Companion::of).apply(builder).build()
}

@EntryDsl
public operator fun <T : GraphTrainableModel> ((Array<Layer>) -> T).invoke(builder: GraphTrainableModelBuilder<T>.() -> Unit): T {
    return GraphTrainableModelBuilder(this).apply(builder).build()
}