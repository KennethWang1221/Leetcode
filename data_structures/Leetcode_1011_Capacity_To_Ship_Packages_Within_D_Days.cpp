#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int shipWithinDays(vector<int>& weights, int days) {
        int l = *max_element(weights.begin(), weights.end());  // The minimum capacity must be at least the largest weight
        int r = 0;
        for (int w : weights) {
            r += w;  // The maximum possible capacity is the sum of all weights
        }
        
        int min_cap = r;

        // Helper function to check if it's possible to ship within the given number of days with the current capacity
        auto canShip = [&](int cap) {
            int ships = 1, curCap = cap;
            for (int w : weights) {
                if (curCap - w < 0) {  // If current capacity is not enough for the current weight
                    ships++;
                    curCap = cap;  // Start a new ship
                }
                curCap -= w;
            }
            return ships <= days;
        };

        // Perform binary search to find the minimum ship capacity
        while (l <= r) {
            int cap = (l + r) / 2;
            if (canShip(cap)) {
                min_cap = min(min_cap, cap);
                r = cap - 1;  // Try smaller capacities
            } else {
                l = cap + 1;  // Try larger capacities
            }
        }

        return min_cap;
    }
};

int main() {
    Solution solution;
    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int days = 5;

    int result = solution.shipWithinDays(weights, days);
    cout << "Minimum capacity: " << result << endl;  // Expected output: 15

    return 0;
}

// g++ -std=c++17 Leetcode_1011_Capacity_To_Ship_Packages_Within_D_Days.cpp -o test