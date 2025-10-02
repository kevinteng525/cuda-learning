#include <iostream>
#include <vector>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int x = 5;
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::cout << "Hello, C++ sample!" << std::endl;
    std::cout << "x = " << x << std::endl;

    int f = factorial(x);
    std::cout << "factorial(" << x << ") = " << f << std::endl;

    int sum = 0;
    for (int i : v) sum += i;
    std::cout << "sum = " << sum << std::endl;

    // Keep program alive briefly to allow debugger attach if needed
    for (int i = 0; i < 3; ++i) {
        std::cout << "tick " << i << std::endl;
    }

    return 0;
}
