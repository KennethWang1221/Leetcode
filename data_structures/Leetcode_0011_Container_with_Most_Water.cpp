#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();
        int l = 0, r = n - 1;
        int res = 0;

        while (l <= r) {
            int h = min(height[l], height[r]);
            int width = r - l;
            int area = h * width;
            res = max(res, area);

            if (height[l] <= height[r]) {
                l++;
            } else {
                r--;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> height = {1, 8, 6, 2, 5, 4, 8, 3, 7};

    int result = solution.maxArea(height);
    cout << "Maximum area: " << result << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0011_Container_with_Most_Water.cpp -o test