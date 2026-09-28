plugins {
    id("com.android.application")
}

val executorchVersion = providers.gradleProperty("executorchVersion").orElse("1.3.1")
val executorchRuntimeLabel = providers.gradleProperty("executorchRuntimeLabel").orElse(executorchVersion)
val executorchAar = providers.gradleProperty("executorchAar").orNull
val benchmarkVersionCode = providers.gradleProperty("benchmarkVersionCode").map(String::toInt).orElse(1)
val benchmarkVersionName = providers.gradleProperty("benchmarkVersionName").orElse("1.0")

android {
    namespace = "org.cropcop.trackc"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.cropcop.trackc.benchmark"
        minSdk = 24
        targetSdk = 36
        versionCode = benchmarkVersionCode.get()
        versionName = benchmarkVersionName.get()
        buildConfigField("String", "EXECUTORCH_VERSION", "\"${executorchRuntimeLabel.get()}\"")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    if (executorchAar == null) {
        implementation("org.pytorch:executorch-android:${executorchVersion.get()}")
    } else {
        implementation(files(executorchAar))
        implementation("com.facebook.fbjni:fbjni:0.7.0")
        implementation("com.facebook.soloader:nativeloader:0.10.5")
    }
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    testImplementation("junit:junit:4.13.2")
}
