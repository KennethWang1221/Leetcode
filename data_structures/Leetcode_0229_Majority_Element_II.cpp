#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int n = nums.size();
        int threshold = n / 3;
        unordered_map<int, int> freq;
        vector<int> res;

        // Count the frequency of each element
        for (int num : nums) {
            freq[num]++;
        }

        // Collect elements whose frequency is greater than n / 3
        for (const auto& [key, value] : freq) {
            if (value > threshold) {
                res.push_back(key);
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {3, 2, 3};
    vector<int> res = solution.majorityElement(nums);

    // Print the result
    for (int num : res) {
        cout << num << " ";
    }
    cout << endl;  // Expected output: 3

    return 0;
}


// g++ -std=c++17 Leetcode_0229_Majority_Element_II.cpp -o test