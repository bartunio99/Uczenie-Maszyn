# main/CMakeLists.txt
idf_component_register(
    SRCS 
        "main.cpp" 
        "l1_structured.cc"
    INCLUDE_DIRS 
        "."
    REQUIRES 
        esp_timer 
        heap 
        esp-tflite-micro
)