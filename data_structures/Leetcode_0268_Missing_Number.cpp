#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int res = nums.size();  // Start with the length of the array
        
        for (int i = 0; i < nums.size(); ++i) {
            res += i - nums[i];  // Sum the index differences
        }
        
        return res;
    }
};

int main() {
    Solution sol;
    vector<int> nums = {3, 0, 1};  // Example input
    int result = sol.missingNumber(nums);
    cout << result << endl;  // Expected output: 2
    
    return 0;
}
// g++ -std=c++17 Leetcode_0268_Missing_Number.cpp -o test