#include <iostream>
#include <vector>
#include <cmath>  // For std::ceil
#include <algorithm>
using namespace std;

class Solution {
public:
    int minEatingSpeed(vector<int>& piles, int h) {
        int l = 1, r = *max_element(piles.begin(), piles.end());
        int res = r;

        while (l <= r) {
            int k = (l + r) / 2;

            long long totalTime = 0;  // Use long long to handle large values
            for (int p : piles) {
                totalTime += ceil((double)p / k);  // Use ceil to simulate the rounding up
            }

            if (totalTime <= h) {
                res = min(res, k);
                r = k - 1;
            } else {
                l = k + 1;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> piles = {3, 6, 7, 11};
    int h = 8;

    int result = solution.minEatingSpeed(piles, h);
    cout << "Minimum Eating Speed: " << result << endl;  // Expected output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0875_KoKo_Eating_Bananas.cpp -o test