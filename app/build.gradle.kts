plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    id("com.google.gms.google-services")
    id("com.chaquo.python") // ✅ Chaquopy 추가
}

android {
    namespace = "com.aicheck"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.aicheck"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("x86_64", "arm64-v8a") // ✅ ONNX용 NDK 설정
        }
    }

    sourceSets {
        getByName("main") {
            manifest.srcFile("src/main/AndroidManifest.xml")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        noCompress += listOf("onnx") // ✅ ONNX 모델 압축 방지
    }
}

chaquopy {
    defaultConfig {
        // ✅ Chaquopy용 파이썬 경로 지정 (Windows 개발 시)
        buildPython("C:\\Users\\SSAFY\\AppData\\Local\\Programs\\Python\\Python38\\python.exe")
        pip {
            install("librosa==0.8.1")
            install("resampy==0.2.2")
            install("numba==0.48.0")
            install("tensorflow==2.1.0")
            install("protobuf==3.20.3")
            install("matplotlib")
            install("numpy")
            install("pillow")
            install("scipy")
        }
    }
}

dependencies {
    // ✅ 너 원래 있던 의존성
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)

    implementation("androidx.biometric:biometric:1.2.0-alpha05")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4")
    implementation("androidx.activity:activity-ktx:1.7.2")
    implementation("androidx.fragment:fragment-ktx:1.6.2")
    implementation("com.google.android.material:material:1.3.0")

    // ✅ FFmpegKit
    implementation("com.arthenica:ffmpeg-kit-full:6.0-2")

    // ✅ Firebase
    implementation(platform("com.google.firebase:firebase-bom:33.11.0"))
    implementation("com.google.firebase:firebase-analytics")
    implementation("com.google.firebase:firebase-messaging-ktx:23.4.1")

    // ✅ TensorFlow Lite & TF-Select-ops
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.13.0")

    // ✅ ONNX Runtime
    implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.15.1")

    // ✅ FFT 처리용
    implementation("com.github.wendykierp:JTransforms:3.1")

    // ✅ Kotlin Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // ✅ 기존 .jar 파일 유지
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar"))))
}
