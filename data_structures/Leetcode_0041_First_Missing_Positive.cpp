#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        vector<int>& A = nums;
        
        // Step 1: Replace negative numbers and zeros with 0
        for (int i = 0; i < A.size(); ++i) {
            if (A[i] < 0) {
                A[i] = 0;
            }
        }

        // Step 2: Mark values in the range [1, n] in the array
        for (int i = 0; i < A.size(); ++i) {
            int val = abs(A[i]);
            if (1 <= val && val <= A.size()) {
                if (A[val - 1] > 0) {
                    A[val - 1] *= -1;  // Mark as visited (negate the number)
                } else if (A[val - 1] == 0) {
                    A[val - 1] = -(A.size() + 1);  // Use a special mark for 0
                }
            }
        }

        // Step 3: Find the first missing positive number
        for (int i = 0; i < A.size(); ++i) {
            if (A[i] >= 0) {
                return i + 1;  // i + 1 is the missing number
            }
        }

        // If no missing number is found, return n + 1
        return A.size() + 1;
    }
};

int main() {
    Solution solution;

    vector<int> nums = {3, 4, -1, 1};
    int res = solution.firstMissingPositive(nums);
    cout << res << endl;  // Expected output: 2

    return 0;
}


// g++ -std=c++17 Leetcode_0041_First_Missing_Positive.cpp -o test