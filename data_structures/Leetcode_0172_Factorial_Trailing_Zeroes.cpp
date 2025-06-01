#include <iostream>

class Solution {
public:
    int trailingZeroes(int n) {
        int count = 0;
        while (n >= 5) {
            n /= 5;
            count += n;
        }
        return count;
    }
};

int main() {
    Solution solution;
    
    std::cout << "Test case 3: " << solution.trailingZeroes(3) << std::endl;   // Output: 0
    std::cout << "Test case 5: " << solution.trailingZeroes(5) << std::endl;   // Output: 1
    std::cout << "Test case 25: " << solution.trailingZeroes(25) << std::endl; // Output: 6
    
    return 0;
}

// g++ -std=c++17 Leetcode_0172_Factorial_Trailing_Zeroes.cpp -o test