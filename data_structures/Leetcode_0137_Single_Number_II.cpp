#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }
        return ones;
    }
};

int main() {
    Solution solution;
    
    vector<int> nums1 = {2, 2, 3, 2};
    cout << "Test case 1: " << solution.singleNumber(nums1) << endl; // Expected: 3
    
    vector<int> nums2 = {0, 1, 0, 1, 0, 1, 99};
    cout << "Test case 2: " << solution.singleNumber(nums2) << endl; // Expected: 99
    
    vector<int> nums3 = {4, 4, 4, 5, 5, 5, 6};
    cout << "Test case 3: " << solution.singleNumber(nums3) << endl; // Expected: 6
    
    return 0;
}

// g++ -std=c++17 Leetcode_0137_Single_Number_II.cpp -o test