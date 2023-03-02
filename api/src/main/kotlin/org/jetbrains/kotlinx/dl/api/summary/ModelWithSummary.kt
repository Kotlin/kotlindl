package org.jetbrains.kotlinx.dl.api.summary

/**
 * Interface for the models for which we can produce meaningful summary.
 */
public interface ModelWithSummary {
    /**
     * Returns model summary.
     *
     * @return model summary
     */
    public fun summary(): ModelSummary
}
