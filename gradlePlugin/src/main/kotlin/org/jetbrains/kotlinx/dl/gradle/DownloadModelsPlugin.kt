/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.gradle

import com.android.build.api.dsl.CommonExtension
import com.android.build.gradle.api.AndroidBasePlugin
import com.android.build.gradle.internal.api.DefaultAndroidSourceDirectorySet
import de.undercouch.gradle.tasks.download.Download
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider
import java.io.File


/**
 * Gradle plugin class which creates and configures a download task for
 * getting pretrained models from the model hub server.
 * This task is executed before the project is built.
 */
class DownloadModelsPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val extension = project.extensions.create("downloadKotlinDLModels", DownloadModelExtension::class.java)
        val taskProvider = project.tasks.register("downloadKotlinDLModels", Download::class.java) { downloadTask ->
            val models = extension.modelTypes.get()
            if (models.isEmpty()) {
                downloadTask.enabled = false
                return@register
            }
            models.forEach { modelType ->
                downloadTask.src(awsS3Url + "/" + modelType.serverPath)
            }

            val sourceDirectory = getResourcesDirectory(project, extension.sourceSet.get())
                ?: "src${File.separator}main${File.separator}res"
            downloadTask.dest(sourceDirectory + File.separator + "raw")

            downloadTask.overwrite(extension.overwrite.get())
        }
        project.tasks.getByName("preBuild").dependsOn(taskProvider)
    }

    private fun getResourcesDirectory(project: Project, sourceSetName: String): String? {
        if (project.plugins.withType(AndroidBasePlugin::class.java).isEmpty()) return null
        val androidExtension = project.extensions.findByType(CommonExtension::class.java) ?: return null
        val sourceSet = androidExtension.sourceSets.findByName(sourceSetName) ?: return null
        val resDirectorySet = sourceSet.res as? DefaultAndroidSourceDirectorySet ?: return null
        return resDirectorySet.srcDirs.first().absolutePath
    }

    companion object {
        private const val awsS3Url = "https://kotlindl.s3.amazonaws.com"
    }
}

/**
 * Extension for configuring the download.
 */
abstract class DownloadModelExtension {
    /**
     * List of model type names to download from the model hub.
     * @see [ModelType]
     */
    abstract val models: ListProperty<String>

    /**
     * List of model types to download from the model hub.
     * @see [ModelType]
     */
    val modelTypes: Provider<List<ModelType>> =
        models.map { namesList -> namesList.mapNotNull { name -> ModelType.values().find { it.name == name } } }

    /**
     * Configures if the previously downloaded file should be overwritten.
     */
    abstract val overwrite: Property<Boolean>

    /**
     * Name of the source set where to download the models to.
     */
    abstract val sourceSet: Property<String>

    init {
        models.convention(listOf())
        overwrite.convention(true)
        sourceSet.convention("main")
    }
}
