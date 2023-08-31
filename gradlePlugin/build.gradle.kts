repositories {
    gradlePluginPortal()
}

plugins {
    kotlin("jvm")
    id("java-gradle-plugin")
    id("maven-publish")
    id("com.gradle.plugin-publish")
}

tasks.compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

dependencies {
    implementation(gradleApi())
    implementation("de.undercouch.download:de.undercouch.download.gradle.plugin:4.1.2")
    implementation("com.android.tools.build:gradle:7.3.1")
}

gradlePlugin {
    plugins {
        create("kotlinDlGradlePlugin") {
            id = "org.jetbrains.kotlinx.kotlin-deeplearning-gradle-plugin"
            implementationClass = "org.jetbrains.kotlinx.dl.gradle.DownloadModelsPlugin"
            displayName = "KotlinDL Gradle Plugin"
            description = "Adds a task for downloading Kotlin DL pretrained models from the model hub"
        }
    }
}

pluginBundle {
    website = "https://github.com/Kotlin/kotlindl"
    vcsUrl = "https://github.com/Kotlin/kotlindl"
    description = "Adds a task for downloading Kotlin DL pretrained models from model hub"
    tags = listOf("kotlin", "deep-learning")
}