#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>
using namespace std;

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int maxArea = 0;
        stack<pair<int, int>> stack;  // Pair: (index, height)

        for (int i = 0; i < heights.size(); i++) {
            int start = i;
            // While there is a bar of height greater than the current one, calculate area
            while (!stack.empty() && stack.top().second > heights[i]) {
                auto [index, height] = stack.top();
                stack.pop();
                maxArea = max(maxArea, height * static_cast<int>(i - index));  // Cast to int for comparison
                start = index;
            }
            stack.push({start, heights[i]});
        }

        // Calculate remaining areas for bars still in the stack
        while (!stack.empty()) {
            auto [index, height] = stack.top();
            stack.pop();
            maxArea = max(maxArea, height * static_cast<int>(heights.size() - index));  // Cast to int
        }

        return maxArea;
    }
};

int main() {
    Solution solution;
    vector<int> heights = {2, 1, 5, 6, 2, 3};

    int result = solution.largestRectangleArea(heights);
    cout << "Largest Rectangle Area: " << result << endl;  // Expected output: 10

    return 0;
}


// g++ -std=c++17 Leetcode_0084_Largest_Rectangle_in_Histogram.cpp -o test