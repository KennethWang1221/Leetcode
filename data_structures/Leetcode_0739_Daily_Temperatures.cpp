#include <iostream>
#include <vector>
#include <stack>
using namespace std;

class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<pair<int, int>> stack; // Stack to store pairs of (temperature, index)
        int n = temperatures.size();
        vector<int> res(n, 0); // Initialize result array with zeros

        for (int i = 0; i < n; i++) {
            // While the stack is not empty and the current temperature is greater than the top of the stack
            while (!stack.empty() && temperatures[i] > stack.top().first) {
                auto [stackTemp, stackIndex] = stack.top();
                stack.pop();
                res[stackIndex] = i - stackIndex; // The difference in days
            }
            stack.push({temperatures[i], i}); // Push the current temperature and its index
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> temperatures = {73, 74, 75, 71, 69, 72, 76, 73};

    vector<int> result = solution.dailyTemperatures(temperatures);

    cout << "Result: ";
    for (int temp : result) {
        cout << temp << " ";
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0739_Daily_Temperatures.cpp -o test