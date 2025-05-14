#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
using namespace std;

class Solution {
public:
    int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
        // Min-heap for projects by capital
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minCapital;
        // Max-heap for projects by profit (negative profits to simulate max-heap)
        priority_queue<int> maxProfit;
        
        // Create a min heap with (capital, profit) pairs
        for (int i = 0; i < capital.size(); ++i) {
            minCapital.push({capital[i], profits[i]});
        }
        
        // Run for k iterations or until no more projects can be taken
        for (int i = 0; i < k; ++i) {
            // Add all projects that can be afforded by the current capital
            while (!minCapital.empty() && minCapital.top().first <= w) {
                int c = minCapital.top().first;
                int p = minCapital.top().second;
                minCapital.pop();
                maxProfit.push(p);  // Push profit into max-heap
            }

            // If no project can be taken, break early
            if (maxProfit.empty()) {
                break;
            }

            // Take the most profitable project
            w += maxProfit.top();
            maxProfit.pop();
        }

        return w;
    }
};

int main() {
    Solution solution;
    int k = 2;
    int w = 0;
    vector<int> profits = {1, 2, 3};
    vector<int> capital = {0, 1, 1};

    int res = solution.findMaximizedCapital(k, w, profits, capital);
    cout << res << endl;  // Expected output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0502_IPO.cpp -o test