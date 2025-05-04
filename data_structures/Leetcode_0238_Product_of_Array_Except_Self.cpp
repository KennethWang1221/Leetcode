#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> res(n, 1);  // Initialize the result vector with 1s

        // Calculate the prefix product (left products)
        int prefix = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = prefix;
            prefix *= nums[i];  // Update prefix product for next element
        }

        // Calculate the postfix product (right products) and multiply it with prefix product
        int postfix = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] *= postfix;
            postfix *= nums[i];  // Update postfix product for next element
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 2, 3, 4};
    vector<int> res = solution.productExceptSelf(nums);

    // Print the result
    for (int num : res) {
        cout << num << " ";
    }
    cout << endl;  // Expected output: 24 12 8 6

    return 0;
}


// g++ -std=c++17 Leetcode_0238_Product_of_Array_Except_Self.cpp -o test