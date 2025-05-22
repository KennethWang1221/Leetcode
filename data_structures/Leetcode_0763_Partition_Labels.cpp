#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

class Solution {
public:
    vector<int> partitionLabels(string s) {
        unordered_map<char, int> lastIndex;
        int n = s.size();

        // Step 1: Record last index of each character
        for (int i = 0; i < n; ++i) {
            lastIndex[s[i]] = i;
        }

        vector<int> res;
        int size = 0;
        int end = 0;

        // Step 2: Traverse string to determine partitions
        for (int i = 0; i < n; ++i) {
            size += 1;
            end = max(end, lastIndex[s[i]]);

            if (i == end) {
                res.push_back(size);
                size = 0;
            }
        }

        return res;
    }
};

// Test Case
int main() {
    Solution sol;
    string s = "ababcbacadefegdehijhklij";
    vector<int> result = sol.partitionLabels(s);

    cout << "[ ";
    for (int val : result) {
        cout << val << " ";
    }
    cout << "]" << endl;

    return 0;
}
// g++ -std=c++11 Leetcode_0763_Partition_Labels.cpp -o test && ./test