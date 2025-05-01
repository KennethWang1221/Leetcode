#include <iostream>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution {
public:
    void reverseString(vector<char>& s) {
        int left = 0, right = s.size() - 1;
        
        while (left < right){
            swap(s[left], s[right]);
            
            left++;
            right--;
        }
    }
};

int main() {
    Solution sol;
    vector<char> input = {'h','e','l','l','o'};
    sol.reverseString(input);
    cout << "Test case1:";
    for (char c: input){
        cout << c << " ";
    }
    cout << endl;
    return 0;
}
// g++ -std=c++17 Leetcode_0344_Reverse_String.cpp -o test