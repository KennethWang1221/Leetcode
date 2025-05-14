#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        // Sort trips based on the start time of each trip
        sort(trips.begin(), trips.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[1] < b[1];
        });

        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;  // Min-heap based on end times
        int curPass = 0;  // Current number of passengers

        for (const auto& trip : trips) {
            int numPass = trip[0], start = trip[1], end = trip[2];

            // Remove passengers who have already been dropped off before the current start time
            while (!minHeap.empty() && minHeap.top().first <= start) {
                curPass -= minHeap.top().second;
                minHeap.pop();
            }

            // Add passengers for the current trip
            curPass += numPass;

            // Check if the current number of passengers exceeds the vehicle capacity
            if (curPass > capacity) {
                return false;
            }

            // Push the trip into the heap with the end time and the number of passengers
            minHeap.push({end, numPass});
        }

        return true;
    }
};

int main() {
    Solution solution;
    vector<vector<int>> trips = {{3, 2, 8}, {4, 4, 6}, {10, 8, 9}};
    int capacity = 11;

    bool result = solution.carPooling(trips, capacity);
    cout << (result ? "True" : "False") << endl;  // Output should be "True" or "False"

    return 0;
}


// g++ -std=c++17 Leetcode_1094_Car_Pooling.cpp -o test