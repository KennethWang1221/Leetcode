#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  // For std::stoi
using namespace std;

class Solution {
public:
    int calPoints(vector<string>& operations) {
        vector<int> res;

        for (const string& op : operations) {
            if (op == "C") {
                res.pop_back();
            } else if (op == "D") {
                int temp = res.back();
                res.push_back(temp * 2);
            } else if (op == "+") {
                int add1 = res.back();
                res.pop_back();
                int add2 = res.back();
                res.push_back(add1);
                res.push_back(add1 + add2);
            } else {
                res.push_back(stoi(op));
            }
        }

        int total = 0;
        for (int score : res) {
            total += score;
        }

        return total;
    }
};

int main() {
    Solution solution;
    vector<string> operations = {"5", "-2", "4", "C", "D", "9", "+", "+"};

    int result = solution.calPoints(operations);
    cout << "Total score: " << result << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0682_Baseball_Game.cpp -o test