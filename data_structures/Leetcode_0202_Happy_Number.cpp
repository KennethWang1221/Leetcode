#include <iostream>
#include <unordered_set>
using namespace std;

class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int> num_set;

        while (n != 1) {
            int sum = 0;
            // Calculate the sum of squares of digits
            while (n > 0) {
                int digit = n % 10;  // Get the last digit
                sum += digit * digit;
                n /= 10;  // Remove the last digit
            }

            // Check if we have already encountered this sum (cycle)
            if (num_set.find(sum) != num_set.end()) {
                return false;  // Cycle detected
            }

            // Add the sum to the set to track it
            num_set.insert(sum);
            n = sum;  // Update n to the sum of squares of digits
        }

        return true;  // The number is happy if we reach 1
    }
};

int main() {
    Solution solution;

    // Test case
    int n = 2;
    bool result = solution.isHappy(n);

    cout << "Is " << n << " a happy number? " << (result ? "Yes" : "No") << endl;  // Expected output: No

    return 0;
}

// g++ -std=c++17 Leetcode_0202_Happy_Number.cpp -o test
// ./test