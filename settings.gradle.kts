pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()

        // ✅ Chaquopy 플러그인용 저장소 추가
        maven {
            name = "ChaquoPlugin"
            url = uri("https://chaquo.com/maven")
        }
    }

    // ✅ Chaquopy 플러그인 선언
    plugins {
        id("com.chaquo.python") version "16.0.0"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") } // ✅ 추가해도 무방
    }
}

rootProject.name = "AICheck"
include(":app")
