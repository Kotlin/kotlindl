project.description = "This module contains the Kotlin API for building, training, and evaluating the Deep Learning models."

plugins {
    kotlin("jvm")
}

tasks.compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

kotlin {
    explicitApiWarning()
}

tasks.register<Jar>("sourcesJar") {
    classifier = "sources"
    from(sourceSets["main"].allSource)
}

artifacts {
    archives(tasks.named("sourcesJar"))
}