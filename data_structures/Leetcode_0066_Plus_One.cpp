#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int n = digits.size();
        int carry = 1;  // Initialize carry to 1 since we're adding 1

        // Traverse the digits from right to left
        for (int i = n - 1; i >= 0; --i) {
            digits[i] += carry;  // Add carry to the current digit
            if (digits[i] == 10) {
                digits[i] = 0;  // If the result is 10, set the digit to 0 and carry over 1
                carry = 1;
            } else {
                carry = 0;  // No carry needed, we can break the loop
                break;
            }
        }

        // If there's a carry left, add it at the beginning of the vector
        if (carry == 1) {
            digits.insert(digits.begin(), 1);
        }

        return digits;
    }
};

int main() {
    Solution solution;

    // Test case
    vector<int> digits = {1, 2, 3};
    vector<int> result = solution.plusOne(digits);

    // Print the result
    cout << "Result: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;  // Expected output: 1 2 4

    return 0;
}

// g++ -std=c++17 Leetcode_0066_Plus_One.cpp -o test
// ./test