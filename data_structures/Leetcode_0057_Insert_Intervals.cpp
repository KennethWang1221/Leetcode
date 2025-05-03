#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> merged = intervals;
        merged.push_back(newInterval);
        sort(merged.begin(), merged.end());
        
        vector<vector<int>> res;
        if (merged.empty()) {
            return res;
        }
        res.push_back(merged[0]);
        
        for (int i = 0; i < merged.size(); ++i) {
            vector<int>& last = res.back();
            int current_start = merged[i][0];
            int current_end = merged[i][1];
            
            if (current_start <= last[1]) {
                // Merge the intervals
                last[1] = max(last[1], current_end);
                last[0] = min(last[0], current_start);
            } else {
                res.push_back(merged[i]);
            }
        }
        
        return res;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> intervals = {{1,2}, {3,5}, {6,7}, {8,10}, {12,16}};
    vector<int> newInterval = {4,8};
    vector<vector<int>> result = sol.insert(intervals, newInterval);
    
    // Print the result
    cout << "[";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << "[" << result[i][0] << "," << result[i][1] << "]";
        if (i != result.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl; // Expected output: [[1,2],[3,10],[12,16]]
    
    return 0;
}

// g++ -std=c++17 Leetcode_0057_Insert_Intervals.cpp -o test