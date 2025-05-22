#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        int cur_distance = 0;  // Current farthest index we can reach
        int ans = 0;           // Number of jumps
        int next_distance = 0; // Farthest index we can reach in the next jump
        
        // Iterate through the array, except the last element (since we don't need to jump from the last element)
        for (int i = 0; i < n - 1; ++i) {
            // Calculate the farthest we can reach from the current position
            next_distance = max(next_distance, i + nums[i]);
            
            // If we've reached the current farthest distance, we need to make a jump
            if (i == cur_distance) {
                cur_distance = next_distance; // Update the current farthest distance
                ans++; // Increment the jump count
            }
        }

        return ans;
    }
};

int main() {
    Solution solution;
    
    // Test case
    vector<int> nums = {2, 3, 1, 1, 4};
    int result = solution.jump(nums);
    
    cout << "Minimum jumps: " << result << endl;  // Expected output: 2

    return 0;
}

// g++ -std=c++17 Leetcode_0045_Jump_Game_II.cpp -o test 