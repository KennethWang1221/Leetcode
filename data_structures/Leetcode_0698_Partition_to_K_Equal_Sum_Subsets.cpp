#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
using namespace std;

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int n = nums.size();
        
        // Calculate the total sum of elements
        int total = 0;
        for (int num : nums) {
            total += num;
        }
        
        // If the total sum is not divisible by k, return false
        if (total % k != 0) {
            return false;
        }
        
        int target = total / k;
        
        // Sort in descending order to optimize the backtracking
        sort(nums.rbegin(), nums.rend());
        
        vector<int> sides(k, 0);  // To track the sum of each subset
        vector<bool> visited(n, false);  // To track the used elements

        return backtrack(nums, 0, sides, visited, target, k);
    }

private:
    bool backtrack(vector<int>& nums, int idx, vector<int>& sides, vector<bool>& visited, int target, int k) {
        // If we've used all numbers, check if all sides are equal to the target sum
        if (idx == nums.size()) {
            for (int i = 0; i < k; ++i) {
                if (sides[i] != target) {
                    return false;
                }
            }
            return true;
        }

        // Try to place the current number in any of the k sides
        for (int i = 0; i < k; ++i) {
            if (visited[idx] || sides[i] + nums[idx] > target) {
                continue;
            }

            visited[idx] = true;
            sides[i] += nums[idx];

            // Recurse to the next number
            if (backtrack(nums, idx + 1, sides, visited, target, k)) {
                return true;
            }

            // Backtrack
            visited[idx] = false;
            sides[i] -= nums[idx];

            // If the current side is empty, then no need to try the next sides
            if (sides[i] == 0) {
                break;
            }
        }

        return false;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {2, 2, 2, 2, 3, 4, 5};
    int k = 4;

    bool result = solution.canPartitionKSubsets(nums, k);
    cout << (result ? "True" : "False") << endl;  // Output: False

    return 0;
}

// g++ -std=c++17 Leetcode_0698_Partition_to_K_Equal_Sum_Subsets.cpp -o test