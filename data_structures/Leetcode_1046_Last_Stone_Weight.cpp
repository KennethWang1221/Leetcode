#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        priority_queue<int> maxheap;  // Max-heap to store stones

        // Push all stones into the max-heap
        for (int s : stones) {
            maxheap.push(s);
        }

        // Process the stones until only one or none are left
        while (maxheap.size() > 1) {
            int n1 = maxheap.top();  // The largest stone
            maxheap.pop();
            int n2 = maxheap.top();  // The second largest stone
            maxheap.pop();

            int val = n1 - n2;  // The result of smashing the two stones

            if (val > 0) {
                maxheap.push(val);  // Push the remaining stone back into the heap
            }
        }

        // Return the remaining stone, or 0 if no stones are left
        return maxheap.empty() ? 0 : maxheap.top();
    }
};

int main() {
    Solution solution;
    vector<int> stones = {2, 7, 4, 1, 8, 1};
    cout << solution.lastStoneWeight(stones) << endl; // Output should be 1
    return 0;
}


// g++ -std=c++17 Leetcode_1046_Last_Stone_Weight.cpp -o test