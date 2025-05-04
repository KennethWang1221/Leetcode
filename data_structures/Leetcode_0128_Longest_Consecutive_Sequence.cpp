#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> numSet(nums.begin(), nums.end());
        int longest = 0;

        for (int n : numSet) {
            // Check if it's the start of a sequence
            if (numSet.find(n - 1) == numSet.end()) {
                int length = 0;
                while (numSet.find(n + length) != numSet.end()) {
                    length++;
                }
                longest = max(longest, length);
            }
        }

        return longest;
    }
};

int main() {
    Solution solution;

    vector<int> nums = {100, 4, 200, 1, 3, 2};
    int res = solution.longestConsecutive(nums);
    cout << res << endl;  // Expected output: 4 (sequence: 1, 2, 3, 4)

    return 0;
}


// g++ -std=c++17 Leetcode_0128_Longest_Consecutive_Sequence.cpp -o test