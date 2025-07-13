#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        if (intervals.empty()){
            return {newInterval};
        }
        
        intervals.push_back(newInterval);
        sort(intervals.begin(), intervals.end());
        int n = intervals.size();
        vector<vector<int>> res = {intervals[0]};

        for (int i = 1; i < n; i++){  // Start from 1 since we added intervals[0]
            vector<int>& prev = res.back(); // We use & to get a reference to the last element in the result vector (res) so that when we modify prev, we are actually modifying the element inside res , not just a copy. If you don’t use & , then prev becomes a copy of the last interval, and any changes you make to prev will not affect the original data in res .
            int prevstart = prev[0];
            int prevend = prev[1];
            int curstart = intervals[i][0];
            int curend = intervals[i][1];
            
            if (curstart <= prevend){
                prev[0] = min(prevstart, curstart);
                prev[1] = max(prevend, curend);
            } else {
                vector<int> temp = {curstart, curend};
                res.push_back(temp);
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
    
    // Test 1: Print the result
    cout << "[";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << "[" << result[i][0] << "," << result[i][1] << "]";
        if (i != result.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl; // Expected output: [[1,2],[3,10],[12,16]]

    // Test 2: 
    vector<vector<int>> intervals_2 = {{1,3}, {6,9}};
    vector<int> newInterval_2 = {2,5};
    vector<vector<int>> results  = {};
    results = sol.insert(intervals_2, newInterval_2);
    for (const auto& res : results){
        for (const auto& item : res){
            cout << item << endl;
        }
    }

    return 0;
}

// g++ -std=c++17 Leetcode_0057_Insert_Intervals.cpp -o test