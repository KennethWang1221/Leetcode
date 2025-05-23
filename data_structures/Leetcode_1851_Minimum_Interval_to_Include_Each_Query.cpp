#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> minInterval(vector<vector<int>>& intervals, vector<int>& queries) {
        int n = intervals.size();
        int m = queries.size();
        
        // Sort intervals by their start time
        sort(intervals.begin(), intervals.end());
        
        // Sort queries and remember the original indices
        vector<pair<int, int>> sortedQueries;
        for (int i = 0; i < m; ++i) {
            sortedQueries.push_back({queries[i], i});
        }
        sort(sortedQueries.begin(), sortedQueries.end());

        vector<int> res(m, -1);  // Result array for the queries
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap; // Min-heap

        int i = 0;
        for (auto& query : sortedQueries) {
            int q = query.first;
            int originalIndex = query.second;

            // Push intervals whose start time is <= current query
            while (i < n && intervals[i][0] <= q) {
                int l = intervals[i][0], r = intervals[i][1];
                minHeap.push({r - l + 1, r});  // Push interval length and its end time
                i++;
            }

            // Remove intervals whose end time is less than the current query
            while (!minHeap.empty() && minHeap.top().second < q) {
                minHeap.pop();
            }

            // Assign the result for the current query
            if (!minHeap.empty()) {
                res[originalIndex] = minHeap.top().first;  // Interval length of the smallest valid interval
            }
        }

        return res;
    }
};

int main() {
    Solution solution;

    // Test case
    vector<vector<int>> intervals = {{1, 4}, {2, 4}, {3, 6}, {4, 4}};
    vector<int> queries = {2, 3, 4, 5};

    vector<int> result = solution.minInterval(intervals, queries);
    
    // Print the result
    for (int r : result) {
        cout << r << " ";
    }
    cout << endl;  // Expected output: 3 3 3 4

    return 0;
}
// g++ Leetcode_1851_Minimum_Interval_to_Include_Each_Query.cpp -o test  && ./test