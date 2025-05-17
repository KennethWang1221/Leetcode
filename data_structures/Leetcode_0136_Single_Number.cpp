
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (int n : nums) {
            res = n ^ res;  // XOR operation
        }
        return res;
    }
};

int main() {
    Solution sol;
    vector<int> nums = {2, 2, 1};
    int result = sol.singleNumber(nums);
    cout << result << endl;  // Output: 1

    return 0;
}


// g++ -std=c++17 Leetcode_0136_Single_Number.cpp -o test