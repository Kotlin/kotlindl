repositories {
    gradlePluginPortal()
}

apply plugin: 'kotlin'
apply plugin: 'java-gradle-plugin'
apply plugin: 'maven-publish'
apply plugin: 'com.gradle.plugin-publish'

compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

dependencies {
    implementation gradleApi()
    //noinspection GradleDependency workaround for https://github.com/Kotlin/kotlindl/issues/516
    implementation "de.undercouch.download:de.undercouch.download.gradle.plugin:4.1.2"
    implementation "com.android.tools.build:gradle:7.3.1"
}

gradlePlugin {
    plugins {
        kotlinDlGradlePlugin {
            id = 'org.jetbrains.kotlinx.kotlin-deeplearning-gradle-plugin'
            implementationClass = 'org.jetbrains.kotlinx.dl.gradle.DownloadModelsPlugin'
            displayName = 'KotlinDL Gradle Plugin'
            description = 'Adds a task for downloading Kotlin DL pretrained models from the model hub'
        }
    }
}

pluginBundle {
    website = 'https://github.com/Kotlin/kotlindl'
    vcsUrl = 'https://github.com/Kotlin/kotlindl'
    description = 'Adds a task for downloading Kotlin DL pretrained models from model hub'
    tags = ['kotlin', 'deep-learning']
}