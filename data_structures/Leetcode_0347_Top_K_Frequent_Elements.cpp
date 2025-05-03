#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freqMap;
        for (int num : nums) {
            freqMap[num]++;
        }
        
        vector<pair<int, int>> freqVec;
        for (auto& entry : freqMap) {
            freqVec.push_back({entry.second, entry.first});
        }
        
        sort(freqVec.begin(), freqVec.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            if (a.first == b.first) {
                return a.second < b.second; // Sort by number ascending if frequencies are equal
            }
            return a.first > b.first; // Sort by frequency descending
        });
        
        vector<int> result;
        for (int i = 0; i < k && i < freqVec.size(); ++i) {
            result.push_back(freqVec[i].second);
        }
        
        return result;
    }
};

int main() {
    Solution sol;
    vector<int> nums = {1,1,1,1,2,2,3,3};
    int k = 2;
    vector<int> res = sol.topKFrequent(nums, k);
    cout << "[";
    for (size_t i = 0; i < res.size(); ++i) {
        cout << res[i];
        if (i != res.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl; // Output: [1,2]
    return 0;
}

// g++ -std=c++17 Leetcode_0347_Top_K_Frequent_Elements.cpp -o test