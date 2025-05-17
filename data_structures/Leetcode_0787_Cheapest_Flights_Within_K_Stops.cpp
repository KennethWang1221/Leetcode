#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>

using namespace std;

class Solution {
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        vector<int> prices(n, INT_MAX);  // prices[i] stores the minimum price to reach node i
        prices[src] = 0;  // Starting point

        // Perform Bellman-Ford relaxation up to k+1 times (for at most k stops)
        for (int i = 0; i <= k; ++i) {
            vector<int> tmpPrices = prices;  // Temporary array to hold updated prices
            for (const auto& flight : flights) {
                int s = flight[0], d = flight[1], p = flight[2];
                if (prices[s] == INT_MAX) {
                    continue;  // If the source city is unreachable, skip
                }
                if (prices[s] + p < tmpPrices[d]) {
                    tmpPrices[d] = prices[s] + p;  // Relax the edge
                }
            }
            prices = tmpPrices;  // Update the prices after considering all edges
        }

        return (prices[dst] == INT_MAX) ? -1 : prices[dst];  // Return the result
    }
};

int main() {
    Solution sol;

    int n = 4;
    vector<vector<int>> flights = {{0, 1, 100}, {1, 2, 100}, {2, 0, 100}, {1, 3, 600}, {2, 3, 200}};
    int src = 0, dst = 3, k = 1;
    cout << sol.findCheapestPrice(n, flights, src, dst, k) << endl;  // Output: 700

    n = 3;
    flights = {{0, 1, 100}, {1, 2, 100}, {0, 2, 500}};
    src = 0, dst = 2, k = 1;
    cout << sol.findCheapestPrice(n, flights, src, dst, k) << endl;  // Output: 200

    return 0;
}


// g++ -std=c++17 Leetcode_0787_Cheapest_Flights_Within_K_Stops.cpp -o test 