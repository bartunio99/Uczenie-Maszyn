#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "l1_structured.h"
//#include "imagenet_32x32.h"  // Twój plik z obrazami i etykietami
#include "cifar10_32x32.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "MOBILENET_BENCH";

// Pomocnicza funkcja do znajdowania argmax
int argmax(const int8_t* data, int size) {
    int max_index = 0;
    int8_t max_value = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    return max_index;
}

// Funkcja benchmarku + dokładność
void run_benchmark(tflite::MicroInterpreter& interpreter) {
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t largest_before = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

    printf("Wolny RAM przed inferencja: %d B, najwiekszy blok: %d B\n",
           free_before, largest_before);

    int correct = 0;
    int num_tests = NUM_TEST_IMAGES; // liczba obrazów z .h
    int64_t total_time_us = 0;

    for (int n = 0; n < num_tests; n++) {
        // Skopiuj obraz z tablicy do tensora
        memcpy(input->data.int8, test_images[n], IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS);

        int64_t start = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Blad podczas inferencji dla obrazu %d\n", n);
            continue;
        }
        int64_t end = esp_timer_get_time();
        total_time_us += (end - start);

        // Pobierz argmax
        int predicted_class = argmax(output->data.int8, 1000);  // ImageNet 1000 klas
        int true_class = test_labels[n];  // zakładamy, że w .h masz uint16_t 0-999

        if (predicted_class == true_class) correct++;
        printf("Obraz %d: przewidziano %d, prawdziwa %d %s\n",
               n, predicted_class, true_class,
               (predicted_class == true_class) ? "[OK]" : "[FAIL]");
    }

    float accuracy = (float)correct / num_tests * 100.0f;
    float avg_time_ms = total_time_us / 1000.0f / num_tests;

    printf("Dokladnosc: %.2f%%\n", accuracy);
    printf("Sredni czas inferencji: %.2f ms\n", avg_time_ms);

    size_t free_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t largest_after = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

    printf("Wolny RAM po inferencja: %d B, najwiekszy blok: %d B\n",
           free_after, largest_after);

    printf("Pamiec zuzyta przez tensor arena: %d B\n", free_before - free_after);
}

extern "C" void app_main() {
    printf("Calkowity wolny RAM: %d B\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    printf("Najwiekszy wolny blok: %d B\n", heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    const tflite::Model* model = tflite::GetModel(l1_structured_tflite);

    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddMaxPool2D();
    resolver.AddConcatenation();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddRelu6();
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddMean();
    resolver.AddTranspose();

    const int arena_size = 160 * 1024;
    uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, arena_size);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("!!! ALOKACJA NIEUDANA !!!\n");
        return;
    }

    run_benchmark(interpreter);

    heap_caps_free(tensor_arena);
}
