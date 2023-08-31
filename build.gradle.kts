val majorVersion: String by project
val minorVersion: String by project

plugins {
    kotlin("multiplatform") version "1.8.21" apply false
    id("com.android.application") apply false
    id("maven-publish")
    id("io.github.gradle-nexus.publish-plugin") version "1.1.0"
    id("com.gradle.plugin-publish") version "1.0.0" apply false
    id("org.jetbrains.dokka") version "1.5.30"
}

val kotlinDLVersion = "$majorVersion.$minorVersion"

allprojects {
    repositories {
        google()
        mavenCentral()
    }

    group = "org.jetbrains.kotlinx"
    version = kotlinDLVersion
}

apply {
    from(project.rootProject.file("gradle/fatJar.gradle"))
    from(project.rootProject.file("gradle/dokka.gradle"))
}

val unpublishedSubprojects = setOf("examples", "gradlePlugin")
subprojects {
    if (name in unpublishedSubprojects) return@subprojects
    apply(from = project.rootProject.file("gradle/publish.gradle"))
}

val sonatypeUser: String? = System.getenv("SONATYPE_USER")
val sonatypePassword: String? = System.getenv("SONATYPE_PASSWORD")
nexusPublishing {
    packageGroup.set(project.group.toString())
    repositories {
        sonatype {
            username.set(sonatypeUser)
            password.set(sonatypePassword)
            repositoryDescription.set("kotlinx.kotlindl staging repository, version: $version")
        }
    }
}

tasks.register("setTeamcityVersion") {
    doLast {
        println("##teamcity[buildNumber '$version']")
    }
}