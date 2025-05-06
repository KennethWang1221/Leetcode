#include <iostream>
#include <string>
using namespace std;

class Solution {
public:
    string mergeAlternately(string word1, string word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        string res = "";

        int n = min(n1, n2);
        int i = 0, j = 0;

        while (n > 0) {
            res += word1[i];
            i++;
            res += word2[j];
            j++;

            n--;
        }

        if (i != n1) {
            res += word1.substr(i);
        }
        if (j != n2) {
            res += word2.substr(j);
        }

        return res;
    }
};

int main() {
    Solution solution;
    string word1 = "cdf";
    string word2 = "a";
    cout << solution.mergeAlternately(word1, word2) << endl;
    return 0;
}
// g++ -std=c++17 Leetcode_1768_Merge_Strings_Alternately.cpp -o test