import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm")
}

group = "com.zaleslaw"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("org.tensorflow:tensorflow:1.15.0")
}

tasks.withType(KotlinCompile::class.java) {
    kotlinOptions.jvmTarget = "1.8"
}