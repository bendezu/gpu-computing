#include <iostream>
#include <amp.h>

using namespace std;
using namespace concurrency;

int main() {
    auto accelerators = accelerator::get_all();
    for (const auto& accel: accelerators) {
        wcout << accel.get_description() << endl;
    }
    return 0;
}
