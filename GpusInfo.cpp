#include "lab1.h"

void gpusInfo() {
    auto accelerators = accelerator::get_all();
    for (const auto& accel : accelerators) {
        wcout << accel.get_description() << endl;
        wcout << "Path: " << accel.get_device_path() << endl;
        wcout << "Memory: " << accel.get_dedicated_memory() << "KB" << endl;
        wcout << "Is debug: " << accel.get_is_debug() << endl;
        wcout << "Is emulated: " << accel.get_is_emulated() << endl << endl;
    }
}

