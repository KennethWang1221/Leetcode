#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  // For std::stoi
using namespace std;

class Solution {
public:
    int calPoints(vector<string>& operations) {
        vector<int> res = {};
        int total = 0;
        int n = operations.size();
        for (int i =0; i < n; i++){
            if (operations[i] == "C"){
                res.pop_back();
            } else if ( operations[i] == "D" && !res.empty()) {
                int temp = res.back();
                res.push_back(temp * 2);
            } else if (operations[i] == "+" && !res.empty()) {
                bool use_method1 = true;
                if (use_method1){
                    int res_size = res.size();
                    res.push_back(res[res_size-1] + res[res_size-2]);
                } else {
                    int add1 = res.back();
                    res.pop_back();
                    int add2 = res.back();
                    res.pop_back();
                    res.push_back(add2);
                    res.push_back(add1);
                    res.push_back(add1 + add2);

                }

            } else {
                res.push_back(stoi(operations[i]));
            }
        }

        for (int i = 0; i < res.size(); i++){
            total += res[i];
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